#include "neuralEconomy.h"
#include "constants.h"

// TODO: include the number of currently available offers as a parameter

namespace neural {


torch::Tensor eigenToTorch(Eigen::ArrayXd eigenArray) {
    auto t = torch::empty(eigenArray.rows());
    float* data = t.data_ptr<float>();

    Eigen::Map<Eigen::ArrayXf> arrayMap(data, t.size(0), 1);
    arrayMap = eigenArray.cast<float>();

    // t.requires_grad_(true);

    return t;
}


Eigen::ArrayXd torchToEigen(torch::Tensor tensor) {
    tensor = tensor.to(torch::kFloat64).contiguous();
    return Eigen::Map<Eigen::ArrayXd>(tensor.data_ptr<double>(), tensor.numel());
}

std::pair<torch::Tensor, torch::Tensor> sample_normal(torch::Tensor params) {
    // std::cout << "params: " << params << std::endl;
    auto mu = params.index({"...", 0});
    auto sigma = torch::exp(params.index({"...", 1}));
    int size = (params.dim() > 1) ? params.size(-2) : 1;
    auto normal_vals = torch::randn(size) * sigma + mu;
    auto log_proba = -0.5 * torch::pow((normal_vals - mu) / sigma, 2) - torch::log(sigma * SQRT2PI);
    // std::cout << "logProba: " << log_proba << std::endl;
    return std::make_pair(normal_vals, log_proba);
}

std::pair<torch::Tensor, torch::Tensor> sample_logitNormal(torch::Tensor params) {
    auto pair = sample_normal(params);
    return std::make_pair(torch::sigmoid(pair.first), pair.second);
}

std::pair<torch::Tensor, torch::Tensor> sample_logNormal(torch::Tensor params) {
    auto pair = sample_normal(params);
    return std::make_pair(torch::exp(pair.first), pair.second);
}


torch::Tensor get_purchase_probas(
    torch::Tensor offerIndices, // dtype = kInt64
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> purchaseNet,
    torch::Tensor encodedOffers
) {
    // get encoded offers
    auto offerEncodings = encodedOffers.index_select(0, offerIndices);

    // transform other inputs to torch Tensors
    auto utilParams_ = eigenToTorch(utilParams);
    auto budget_ = torch::tensor({budget});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory);

    // plug into purchaseNet to get probas
    return purchaseNet->forward(offerEncodings, utilParams_, budget_, labor_, inventory_);
}


torch::Tensor get_job_probas(
    torch::Tensor offerIndices, // dtype = kInt64
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> laborSearchNet,
    torch::Tensor encodedJobOffers
) {
    // get encoded offers
    auto jobOfferEncodings = encodedJobOffers.index_select(0, offerIndices);

    // transform other inputs to torch Tensors
    auto utilParams_ = eigenToTorch(utilParams);
    auto money_ = torch::tensor({money});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory);

    // plug into purchaseNet to get probas
    return laborSearchNet->forward(jobOfferEncodings, utilParams_, money_, labor_, inventory_);
}



DecisionNetHandler::DecisionNetHandler(
    std::shared_ptr<NeuralEconomy> economy,
    std::shared_ptr<OfferEncoder> offerEncoder,
    std::shared_ptr<OfferEncoder> jobOfferEncoder,
    std::shared_ptr<PurchaseNet> purchaseNet,
    std::shared_ptr<PurchaseNet> firmPurchaseNet,
    std::shared_ptr<PurchaseNet> laborSearchNet,
    std::shared_ptr<ConsumptionNet> consumptionNet,
    std::shared_ptr<ConsumptionNet> productionNet,
    std::shared_ptr<OfferNet> offerNet,
    std::shared_ptr<JobOfferNet> jobOfferNet,
    std::shared_ptr<ValueNet> valueNet,
    std::shared_ptr<ValueNet> firmValueNet
) : economy(economy),
    offerEncoder(offerEncoder),
    jobOfferEncoder(jobOfferEncoder),
    purchaseNet(purchaseNet),
    firmPurchaseNet(firmPurchaseNet),
    laborSearchNet(laborSearchNet),
    consumptionNet(consumptionNet),
    productionNet(productionNet),
    offerNet(offerNet),
    jobOfferNet(jobOfferNet),
    valueNet(valueNet),
    firmValueNet(firmValueNet)
{
    time_step();
};


DecisionNetHandler::DecisionNetHandler(std::shared_ptr<NeuralEconomy> economy) : economy(economy) {
    int numAgents = economy->get_maxAgents();
    int numGoods = economy->get_numGoods();
    // NOTE: numUtilParams assumes persons have CES utility functions,
    // with goods + labor + tfp + elasticity as params
    // assumes firms have same number of params in their production functs
    int numUtilParams = numGoods + 3;
    int numProdFuncParams = numUtilParams * numGoods;

    offerEncoder = std::make_shared<OfferEncoder>(
        DEFAULT_STACK_SIZE,
        numAgents + numGoods + 1,
        DEFAULT_HIDDEN_SIZE,
        DEFAULT_ENCODING_SIZE
    );
    jobOfferEncoder = std::make_shared<OfferEncoder>(
        DEFAULT_STACK_SIZE,
        numAgents + 2,
        DEFAULT_HIDDEN_SIZE,
        DEFAULT_ENCODING_SIZE
    );
    purchaseNet = std::make_shared<PurchaseNet>(
        offerEncoder,
        numUtilParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    firmPurchaseNet = std::make_shared<PurchaseNet>(
        offerEncoder,
        numProdFuncParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    laborSearchNet = std::make_shared<PurchaseNet>(
        jobOfferEncoder,
        numUtilParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    consumptionNet = std::make_shared<ConsumptionNet>(
        numUtilParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    productionNet = std::make_shared<ConsumptionNet>(
        numProdFuncParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    offerNet = std::make_shared<OfferNet>(
        offerEncoder,
        numProdFuncParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    jobOfferNet = std::make_shared<JobOfferNet>(
        jobOfferEncoder,
        numProdFuncParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    valueNet = std::make_shared<ValueNet>(
        offerEncoder,
        jobOfferEncoder,
        numUtilParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    firmValueNet = std::make_shared<ValueNet>(
        offerEncoder,
        jobOfferEncoder,
        numProdFuncParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );

    time_step();
}


void DecisionNetHandler::update_encodedOffers() {
    offers = economy->get_market();
    unsigned int numOffers = offers.size();

    // NOTE: if the simulation is going to add more agents later,
    // then the size of this should be set to MORE than totalAgents
    torch::Tensor offerers = torch::zeros(
        {numOffers, economy->get_maxAgents()}
    );
    torch::Tensor goods = torch::empty(
        {numOffers, economy->get_numGoods()}
    );
    torch::Tensor prices = torch::empty({numOffers, 1});

    for (int i = 0; i < numOffers; i++) {
        auto offer = offers[i];
        // set offerer
        offerers[i][economy->get_id_for_agent(offer->offerer)] = 1;
        // set goods
        goods[i] = eigenToTorch(offer->quantities).squeeze(-1);
        // set prices
        prices[i] = offer->price;
    }
    auto inputFeatures = torch::cat({goods, prices, offerers}, 1);

    encodedOffers = offerEncoder->forward(inputFeatures);
    numEncodedOffers = numOffers;
}


void DecisionNetHandler::update_encodedJobOffers() {
    jobOffers = economy->get_jobMarket();
    unsigned int numOffers = jobOffers.size();

    // NOTE: this is inefficient, since only firms make job offers...
    torch::Tensor offerers = torch::zeros({numOffers, economy->get_maxAgents()});
    torch::Tensor labors = torch::empty({numOffers, 1});
    torch::Tensor wages = torch::empty({numOffers, 1});

    for (int i = 0; i < numOffers; i++) {
        auto jobOffer = jobOffers[i];
        offerers[i][economy->get_id_for_agent(jobOffer->offerer)] = 1;
        labors[i] = jobOffer->labor;
        wages[i] = jobOffer->wage;
    }
    auto inputFeatures = torch::cat({labors, wages, offerers}, 1);

    encodedJobOffers = jobOfferEncoder->forward(inputFeatures);
    numEncodedJobOffers = numOffers;
}


void DecisionNetHandler::push_back_memory() {
    // Need to keep history of log probas for each net
    // This is for training with advantage actor-critic algorithm
    purchaseNetLogProba.push_back(MapTensor());
    firmPurchaseNetLogProba.push_back(MapTensor());
    laborSearchNetLogProba.push_back(MapTensor());
    consumptionNetLogProba.push_back(MapTensor());
    productionNetLogProba.push_back(MapTensor());
    offerNetLogProba.push_back(MapTensor());
    jobOfferNetLogProba.push_back(MapTensor());
    // Also need predicted values and actual rewards
    values.push_back(MapTensor());
    rewards.push_back(MapTensor());
}


void DecisionNetHandler::time_step() {
    update_encodedOffers();
    update_encodedJobOffers();
    if (time >= 0) {
        push_back_memory();
    }
    time++;
}

void DecisionNetHandler::synchronize_time(std::shared_ptr<Agent> caller) {
    std::lock_guard<std::mutex> lock(myMutex);
    if (caller->get_time() > time) {
        time_step();
    }
}

void DecisionNetHandler::reset(std::shared_ptr<NeuralEconomy> newEconomy) {
    economy = newEconomy;
    time = -1;

    purchaseNetLogProba = {};
    firmPurchaseNetLogProba = {};
    laborSearchNetLogProba = {};
    consumptionNetLogProba = {};
    productionNetLogProba = {};
    offerNetLogProba = {};
    jobOfferNetLogProba = {};
    values = {};
    rewards = {};

    time_step();
}

torch::Tensor DecisionNetHandler::generate_offerIndices() {
    if (numEncodedOffers == 0) {
        return torch::tensor({}, torch::dtype(torch::kInt64));
    }
    // get random indices of offers to consider
    return torch::randint(
        0, numEncodedOffers, purchaseNet->offerEncoder->stackSize, torch::dtype(torch::kInt64)
    );
}

torch::Tensor DecisionNetHandler::generate_jobOfferIndices() {
    if (numEncodedJobOffers == 0) {
        return torch::tensor({}, torch::dtype(torch::kInt64));
    }
    // get random indices of offers to consider
    return torch::randint(
        0, numEncodedJobOffers, laborSearchNet->offerEncoder->stackSize, torch::dtype(torch::kInt64)
    );
}

torch::Tensor DecisionNetHandler::firm_generate_offerIndices() {
    if (numEncodedOffers == 0) {
        return torch::tensor({}, torch::dtype(torch::kInt64));
    }
    // get random indices of offers to consider
    return torch::randint(
        0, numEncodedOffers, firmPurchaseNet->offerEncoder->stackSize, torch::dtype(torch::kInt64)
    );
}

torch::Tensor DecisionNetHandler::firm_generate_jobOfferIndices() {
    if (numEncodedJobOffers == 0) {
        return torch::tensor({}, torch::dtype(torch::kInt64));
    }
    // get random indices of offers to consider
    return torch::randint(
        0, numEncodedJobOffers, jobOfferNet->offerEncoder->stackSize, torch::dtype(torch::kInt64)
    );
}


std::pair<std::vector<Order<Offer>>, torch::Tensor> DecisionNetHandler::create_offer_requests(
    const torch::Tensor& offerIndices, // dtype = kInt64
    const torch::Tensor& purchase_probas
) {
    auto to_purchase = (torch::rand(purchase_probas.sizes()) < purchase_probas);

    std::vector<Order<Offer>> toRequest;
    auto logProba = torch::tensor(0.0);
    for (int i = 0; i < to_purchase.size(0); i++) {
        if (to_purchase[i].item<bool>()) {
            auto offer = offers[offerIndices[i].item<int>()];
            toRequest.push_back(Order<Offer>(offer, 1));
            logProba = logProba + torch::log(purchase_probas[i]);
        }
        else {
            logProba = logProba + torch::log(1 - purchase_probas[i]);
        }
    }
    return std::make_pair(toRequest, logProba);
}


std::vector<Order<Offer>> DecisionNetHandler::get_offers_to_request(
    std::shared_ptr<Agent> caller,
    const torch::Tensor& offerIndices,
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    if (offerIndices.size(0) == 0) {
        std::lock_guard<std::mutex> lock(purchaseNetMutex);
        purchaseNetLogProba[time-1][caller] = torch::tensor(0.0);
        return {};
    }
    // std::cout << "using purchaseNet" << std::endl;
    auto probas = get_purchase_probas(
        offerIndices, utilParams, budget, labor, inventory, purchaseNet, encodedOffers
    );

    auto request_proba_pair = create_offer_requests(offerIndices, probas);
    // std::cout << "Recording logProba at time " << time << " for agent " << caller << std::endl;
    {
        std::lock_guard<std::mutex> lock(purchaseNetMutex);
        purchaseNetLogProba[time-1][caller] = request_proba_pair.second;
    }
    return request_proba_pair.first;
}


std::vector<Order<Offer>> DecisionNetHandler::firm_get_offers_to_request(
    std::shared_ptr<Agent> caller,
    const torch::Tensor& offerIndices,
    const Eigen::ArrayXd& prodFuncParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    if (offerIndices.size(0) == 0) {
        std::lock_guard<std::mutex> lock(firmPurchaseNetMutex);
        firmPurchaseNetLogProba[time-1][caller] = torch::tensor(0.0);
        return {};
    }
    // std::cout << "using firmPurchaseNet" << std::endl;
    auto probas = get_purchase_probas(
        offerIndices, prodFuncParams, budget, labor, inventory, firmPurchaseNet, encodedOffers
    );

    auto request_proba_pair = create_offer_requests(offerIndices, probas);
    {
        std::lock_guard<std::mutex> lock(firmPurchaseNetMutex);
        firmPurchaseNetLogProba[time-1][caller] = request_proba_pair.second;
    }
    return request_proba_pair.first;
}


std::pair<std::vector<Order<JobOffer>>, torch::Tensor> DecisionNetHandler::create_joboffer_requests(
    const torch::Tensor& offerIndices, // dtype = kInt64
    const torch::Tensor& job_probas
) {
    auto to_take = (torch::rand(job_probas.sizes()) < job_probas);

    std::vector<Order<JobOffer>> toRequest;
    auto logProba = torch::tensor(0.0);
    for (int i = 0; i < to_take.size(0); i++) {
        if (to_take[i].item<bool>()) {
            auto jobOffer = jobOffers[offerIndices[i].item<int>()];
            toRequest.push_back(Order<JobOffer>(jobOffer, 1));
            logProba = logProba + torch::log(job_probas[i]);
        }
        else {
            logProba = logProba + torch::log(1 - job_probas[i]);
        }
    }
    return std::make_pair(toRequest, logProba);
}


std::vector<Order<JobOffer>> DecisionNetHandler::get_joboffers_to_request(
    std::shared_ptr<Agent> caller,
    const torch::Tensor& jobOfferIndices,
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    if (jobOfferIndices.size(0) == 0) {
        std::lock_guard<std::mutex> lock(laborSearchNetMutex);
        laborSearchNetLogProba[time-1][caller] = torch::tensor(0.0); // do I need requires_grad(true) here?
        return {};
    }

    // std::cout << "using laborSearchNet" << std::endl;
    auto probas = get_job_probas(
        jobOfferIndices, utilParams, money, labor, inventory, laborSearchNet, encodedJobOffers
    );

    auto request_proba_pair = create_joboffer_requests(jobOfferIndices, probas);
    {
        std::lock_guard<std::mutex> lock(laborSearchNetMutex);
        laborSearchNetLogProba[time-1][caller] = request_proba_pair.second;
    }
    return request_proba_pair.first;
}


Eigen::ArrayXd DecisionNetHandler::get_consumption_proportions(
    std::shared_ptr<Agent> caller,
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    // std::cout << "using consumptionNet" << std::endl;
    auto utilParams_ = eigenToTorch(utilParams);
    auto money_ = torch::tensor({money});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory);
    auto consumption_pair = sample_logitNormal(
        consumptionNet->forward(utilParams_, money_, labor_, inventory_)
    );
    {
        std::lock_guard<std::mutex> lock(consumptionNetMutex);
        consumptionNetLogProba[time-1][caller] = torch::sum(consumption_pair.second);
    }
    return torchToEigen(consumption_pair.first);
}


Eigen::ArrayXd DecisionNetHandler::get_production_proportions(
    std::shared_ptr<Agent> caller,
    const Eigen::ArrayXd& prodFuncParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    // std::cout << "using productionNet" << std::endl;
    auto prodFuncParams_ = eigenToTorch(prodFuncParams);
    auto money_ = torch::tensor({money});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory);
    auto production_pair = sample_logitNormal(
        productionNet->forward(prodFuncParams_, money_, labor_, inventory_)
    );
    {
        std::lock_guard<std::mutex> lock(productionNetMutex);
        productionNetLogProba[time-1][caller] = torch::sum(production_pair.second);
    }
    return torchToEigen(production_pair.first);
}


torch::Tensor DecisionNetHandler::getEncodedOffersFromIndices(
    const torch::Tensor& offerIndices
) {
    if (offerIndices.size(0) == 0) {
        // return a tensor of zeros of the appropriate shape, meaning no offers currently on market
        return torch::zeros({offerEncoder->stackSize, offerEncoder->encodingSize});
    }
    else {
        return encodedOffers.index_select(0, offerIndices);
    }
}


torch::Tensor DecisionNetHandler::getEncodedJobOffersFromIndices(
    const torch::Tensor& offerIndices
) {
    if (offerIndices.size(0) == 0) {
        // return a tensor of zeros of the appropriate shape, meaning no offers currently on market
        return torch::zeros({jobOfferEncoder->stackSize, jobOfferEncoder->encodingSize});
    }
    else {
        return encodedJobOffers.index_select(0, offerIndices);
    }
}


std::pair<Eigen::ArrayXd, Eigen::ArrayXd> DecisionNetHandler::choose_offers(
    std::shared_ptr<Agent> caller,
    const torch::Tensor& offerIndices,
    const Eigen::ArrayXd& prodFuncParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    // std::cout << "using offerNet" << std::endl;
    auto encodedOffers = getEncodedOffersFromIndices(offerIndices);
    auto netOutput = offerNet->forward(
        encodedOffers,
        eigenToTorch(prodFuncParams),
        torch::tensor({money}),
        torch::tensor({labor}),
        eigenToTorch(inventory)
    );

    auto amounts_params = netOutput.index({"...", torch::tensor({0, 1})});
    auto amount_pair = sample_logitNormal(amounts_params);
    auto amounts = torchToEigen(amount_pair.first) * inventory;

    auto prices_params = netOutput.index({"...", torch::tensor({2, 3})});
    auto price_pair = sample_logNormal(prices_params);
    auto prices = torchToEigen(price_pair.first);

    {
        std::lock_guard<std::mutex> lock(offerNetMutex);
        offerNetLogProba[time-1][caller] = (
            torch::sum(amount_pair.second)
            + torch::sum(price_pair.second)
        );
    }

    return std::make_pair(amounts, prices);
}


std::pair<double, double> DecisionNetHandler::choose_job_offers(
    std::shared_ptr<Agent> caller,
    const torch::Tensor& offerIndices,
    const Eigen::ArrayXd& prodFuncParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    // std::cout << "using jobOfferNet" << std::endl;
    auto encodedOffers = getEncodedJobOffersFromIndices(offerIndices);
    auto netOutput = jobOfferNet->forward(
        encodedOffers,
        eigenToTorch(prodFuncParams),
        torch::tensor({money}),
        torch::tensor({labor}),
        eigenToTorch(inventory)
    );

    auto labor_params = netOutput.index({"...", torch::tensor({0, 1})});
    auto labor_pair = sample_logNormal(labor_params);
    double totalLabor = labor_pair.first.item<double>();

    auto wage_params = netOutput.index({"...", torch::tensor({2, 3})});
    auto wage_pair = sample_logNormal(wage_params);
    double wage = wage_pair.first.item<double>();
    // clip wage to avoid inf values
    if (wage > constants::largeNumber) {
        wage = constants::largeNumber;
        pprint(2, "Note: Clipped wage to " + std::to_string(constants::largeNumber));
    }

    {
        std::lock_guard<std::mutex> lock(jobOfferNetMutex);
        jobOfferNetLogProba[time-1][caller] = (labor_pair.second + wage_pair.second)[0];
    }

    return std::make_pair(totalLabor, wage);
}


void DecisionNetHandler::record_value(
    std::shared_ptr<Agent> caller,
    const torch::Tensor& offerIndices,
    const torch::Tensor& jobOfferIndices,
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto offerEncodings = getEncodedOffersFromIndices(offerIndices);
    auto jobOfferEncodings = getEncodedJobOffersFromIndices(jobOfferIndices);
    auto utilParams_ = eigenToTorch(utilParams);
    auto money_ = torch::tensor({money});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory);
    {
        std::lock_guard<std::mutex> lock(valueNetMutex);
        values[time-1][caller] = valueNet->forward(
            offerEncodings,
            jobOfferEncodings,
            utilParams_,
            money_,
            labor_,
            inventory_
        );
    }
}


void DecisionNetHandler::firm_record_value(
    std::shared_ptr<Agent> caller,
    const torch::Tensor& offerIndices,
    const torch::Tensor& jobOfferIndices,
    const Eigen::ArrayXd& prodFuncParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto offerEncodings = getEncodedOffersFromIndices(offerIndices);
    auto jobOfferEncodings = getEncodedJobOffersFromIndices(jobOfferIndices);
    auto prodFuncParams_ = eigenToTorch(prodFuncParams);
    auto money_ = torch::tensor({money});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory);
    {
        std::lock_guard<std::mutex> lock(firmValueNetMutex);
        values[time-1][caller] = firmValueNet->forward(
            offerEncodings,
            jobOfferEncodings,
            prodFuncParams_,
            money_,
            labor_,
            inventory_
        );
    }
}


void DecisionNetHandler::record_reward(
    std::shared_ptr<Agent> caller,
    double reward
) {
    {
        std::lock_guard<std::mutex> lock(myMutex);
        rewards[time-1][caller] = torch::tensor(reward);
    }
}

void DecisionNetHandler::record_reward(
    std::shared_ptr<Agent> caller,
    double reward,
    int offset
) {
    {
        std::lock_guard<std::mutex> lock(myMutex);
        rewards[time - 1 - offset][caller] = torch::tensor(reward);
    }
}


void DecisionNetHandler::save_models() {
    torch::save(offerEncoder, "offerEncoder.pt");
    torch::save(jobOfferEncoder, "jobOfferEncoder.pt");
	torch::save(purchaseNet, "purchaseNet.pt");
    torch::save(firmPurchaseNet, "firmPurchaseNet.pt");
    torch::save(laborSearchNet, "laborSearchNet.pt");
    torch::save(consumptionNet, "consumptionNet.pt");
    torch::save(productionNet, "productionNet.pt");
    torch::save(offerNet, "offerNet.pt");
    torch::save(jobOfferNet, "jobOfferNet.pt");
    torch::save(valueNet, "valueNet.pt");
    torch::save(firmValueNet, "firmValueNet.pt");
}


void DecisionNetHandler::load_models() {
    torch::load(offerEncoder, "offerEncoder.pt");
    torch::load(jobOfferEncoder, "jobOfferEncoder.pt");
	torch::load(purchaseNet, "purchaseNet.pt");
    torch::load(firmPurchaseNet, "firmPurchaseNet.pt");
    torch::load(laborSearchNet, "laborSearchNet.pt");
    torch::load(consumptionNet, "consumptionNet.pt");
    torch::load(productionNet, "productionNet.pt");
    torch::load(offerNet, "offerNet.pt");
    torch::load(jobOfferNet, "jobOfferNet.pt");
    torch::load(valueNet, "valueNet.pt");
    torch::load(firmValueNet, "firmValueNet.pt");
}


} // namespace neural
