#define _USE_MATH_DEFINES
#include <cmath>
#include "decisionNetHandler.h"

// TODO: include the number of currently available offers as a parameter

namespace neural {

const double SQRT2PI = 2 / (M_2_SQRTPI * M_SQRT1_2);

const int DEFAULT_STACK_SIZE = 10;
const int DEFAULT_ENCODING_SIZE = 10;
const int DEFAULT_HIDDEN_SIZE = 30;


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
    auto mu = params.index({"...", 0});
    auto sigma = params.index({"...", 1});
    int size = (params.dim() > 1) ? params.size(-2) : 1;
    auto normal_vals = torch::randn(size) * sigma + mu;
    auto log_proba = -0.5 * torch::pow((normal_vals - mu) / sigma, 2) - torch::log(sigma * SQRT2PI);
    return std::make_pair(normal_vals, log_proba);
}

std::pair<torch::Tensor, torch::Tensor> sample_sigmoidNormal(torch::Tensor params) {
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
    NeuralEconomy* economy,
    std::shared_ptr<OfferEncoder> offerEncoder,
    std::shared_ptr<OfferEncoder> jobOfferEncoder,
    std::shared_ptr<PurchaseNet> purchaseNet,
    std::shared_ptr<PurchaseNet> firmPurchaseNet,
    std::shared_ptr<PurchaseNet> laborSearchNet,
    std::shared_ptr<ConsumptionNet> consumptionNet,
    std::shared_ptr<ConsumptionNet> productionNet,
    std::shared_ptr<OfferNet> offerNet,
    std::shared_ptr<JobOfferNet> jobOfferNet
) : economy(economy),
    offerEncoder(offerEncoder),
    jobOfferEncoder(jobOfferEncoder),
    purchaseNet(purchaseNet),
    firmPurchaseNet(firmPurchaseNet),
    laborSearchNet(laborSearchNet),
    consumptionNet(consumptionNet),
    productionNet(productionNet),
    offerNet(offerNet),
    jobOfferNet(jobOfferNet)
{
    assert(offerNet->stackSize == firmPurchaseNet->stackSize);
    time_step();
};


DecisionNetHandler::DecisionNetHandler(NeuralEconomy* economy) : economy(economy) {
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
        DEFAULT_ENCODING_SIZE,
        DEFAULT_STACK_SIZE,
        numUtilParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    firmPurchaseNet = std::make_shared<PurchaseNet>(
        DEFAULT_ENCODING_SIZE,
        DEFAULT_STACK_SIZE,
        numProdFuncParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    laborSearchNet = std::make_shared<PurchaseNet>(
        DEFAULT_ENCODING_SIZE,
        DEFAULT_STACK_SIZE,
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
        DEFAULT_ENCODING_SIZE,
        DEFAULT_STACK_SIZE,
        numProdFuncParams,
        numGoods,
        DEFAULT_HIDDEN_SIZE
    );
    jobOfferNet = std::make_shared<JobOfferNet>(
        DEFAULT_ENCODING_SIZE,
        DEFAULT_STACK_SIZE,
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


void DecisionNetHandler::push_back_logProbas() {
    // Need to keep history of log probas for each net
    // This is for training with advantage actor-critic algorithm
    offerEncoderLogProba.push_back(torch::tensor(0.0));
    jobOfferEncoderLogProba.push_back(torch::tensor(0.0));
    purchaseNetLogProba.push_back(torch::tensor(0.0));
    firmPurchaseNetLogProba.push_back(torch::tensor(0.0));
    laborSearchNetLogProba.push_back(torch::tensor(0.0));
    consumptionNetLogProba.push_back(torch::tensor(0.0));
    productionNetLogProba.push_back(torch::tensor(0.0));
    offerNetLogProba.push_back(torch::tensor(0.0));
    jobOfferNetLogProba.push_back(torch::tensor(0.0));
}


void DecisionNetHandler::time_step() {
    std::lock_guard<std::mutex> lock(myMutex);
    update_encodedOffers();
    update_encodedJobOffers();
    push_back_logProbas();
    time++;
}

torch::Tensor DecisionNetHandler::generate_offerIndices() {
    if (numEncodedOffers == 0) {
        return torch::tensor({}, torch::dtype(torch::kInt64));
    }
    // get random indices of offers to consider
    return torch::randint(
        0, numEncodedOffers, purchaseNet->stackSize, torch::dtype(torch::kInt64)
    );
}

torch::Tensor DecisionNetHandler::generate_jobOfferIndices() {
    if (numEncodedJobOffers == 0) {
        return torch::tensor({}, torch::dtype(torch::kInt64));
    }
    // get random indices of offers to consider
    return torch::randint(
        0, numEncodedJobOffers, laborSearchNet->stackSize, torch::dtype(torch::kInt64)
    );
}

torch::Tensor DecisionNetHandler::firm_generate_offerIndices() {
    if (numEncodedOffers == 0) {
        return torch::tensor({}, torch::dtype(torch::kInt64));
    }
    // get random indices of offers to consider
    return torch::randint(
        0, numEncodedOffers, firmPurchaseNet->stackSize, torch::dtype(torch::kInt64)
    );
}

torch::Tensor DecisionNetHandler::firm_generate_jobOfferIndices() {
    if (numEncodedJobOffers == 0) {
        return torch::tensor({}, torch::dtype(torch::kInt64));
    }
    // get random indices of offers to consider
    return torch::randint(
        0, numEncodedJobOffers, jobOfferNet->stackSize, torch::dtype(torch::kInt64)
    );
}


std::vector<Order<Offer>> DecisionNetHandler::create_offer_requests(
    const torch::Tensor& offerIndices, // dtype = kInt64
    const torch::Tensor& purchase_probas,
    std::vector<torch::Tensor>& logProba
) {
    auto to_purchase = (torch::rand(purchase_probas.sizes()) < purchase_probas);

    std::vector<Order<Offer>> toRequest;
    for (int i = 0; i < to_purchase.size(0); i++) {
        if (to_purchase[i].item<bool>()) {
            auto offer = offers[offerIndices[i].item<int>()];
            toRequest.push_back(Order<Offer>(offer, 1));
            logProba[time] += torch::log(purchase_probas[i]);
        }
        else {
            logProba[time] += torch::log(1 - purchase_probas[i]);
        }
    }
    return toRequest;
}


std::vector<Order<Offer>> DecisionNetHandler::get_offers_to_request(
    const torch::Tensor& offerIndices,
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto probas = get_purchase_probas(
        offerIndices, utilParams, budget, labor, inventory, purchaseNet, encodedOffers
    );

    return create_offer_requests(offerIndices, probas, purchaseNetLogProba);
}


std::vector<Order<Offer>> DecisionNetHandler::firm_get_offers_to_request(
    const torch::Tensor& offerIndices,
    const Eigen::ArrayXd& prodFuncParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto probas = get_purchase_probas(
        offerIndices, prodFuncParams, budget, labor, inventory, firmPurchaseNet, encodedOffers
    );

    return create_offer_requests(offerIndices, probas, firmPurchaseNetLogProba);
}


std::vector<Order<JobOffer>> DecisionNetHandler::create_joboffer_requests(
    const torch::Tensor& offerIndices, // dtype = kInt64
    const torch::Tensor& job_probas,
    std::vector<torch::Tensor>& logProba
) {
    auto to_take = (torch::rand(job_probas.sizes()) < job_probas);

    std::vector<Order<JobOffer>> toRequest;
    for (int i = 0; i < to_take.size(0); i++) {
        if (to_take[i].item<bool>()) {
            auto jobOffer = jobOffers[offerIndices[i].item<int>()];
            toRequest.push_back(Order<JobOffer>(jobOffer, 1));
            logProba[time] += torch::log(job_probas[i]);
        }
        else {
            logProba[time] += torch::log(1 - job_probas[i]);
        }
    }
    return toRequest;
}


std::vector<Order<JobOffer>> DecisionNetHandler::get_joboffers_to_request(
    const torch::Tensor& jobOfferIndices,
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto probas = get_job_probas(
        jobOfferIndices, utilParams, money, labor, inventory, laborSearchNet, encodedJobOffers
    );

    return create_joboffer_requests(jobOfferIndices, probas, laborSearchNetLogProba);
}


Eigen::ArrayXd DecisionNetHandler::get_consumption_proportions(
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto utilParams_ = eigenToTorch(utilParams);
    auto money_ = torch::tensor({money});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory);
    auto consumption_pair = sample_sigmoidNormal(
        consumptionNet->forward(utilParams_, money_, labor_, inventory_)
    );
    consumptionNetLogProba[time] += torch::sum(consumption_pair.second);
    return torchToEigen(consumption_pair.first);
}


Eigen::ArrayXd DecisionNetHandler::get_production_proportions(
    const Eigen::ArrayXd& prodFuncParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto prodFuncParams_ = eigenToTorch(prodFuncParams);
    auto money_ = torch::tensor({money});
    auto labor_ = torch::tensor({labor});
    auto inventory_ = eigenToTorch(inventory);
    auto production_pair = sample_sigmoidNormal(
        productionNet->forward(prodFuncParams_, money_, labor_, inventory_)
    );
    productionNetLogProba[time] += torch::sum(production_pair.second);
    return torchToEigen(production_pair.first);
}


torch::Tensor DecisionNetHandler::getEncodedOffersForOfferCreation(
    const torch::Tensor& offerIndices
) {
    if (offerIndices.size(0) == 0) {
        // return a tensor of zeros of the appropriate shape, meaning no offers currently on market
        return torch::zeros({offerNet->stackSize, offerEncoder->encodingSize});
    }
    else {
        return encodedOffers.index_select(0, offerIndices);
    }
}


torch::Tensor DecisionNetHandler::getEncodedOffersForJobOfferCreation(
    const torch::Tensor& offerIndices
) {
    if (offerIndices.size(0) == 0) {
        // return a tensor of zeros of the appropriate shape, meaning no offers currently on market
        return torch::zeros({jobOfferNet->stackSize, jobOfferEncoder->encodingSize});
    }
    else {
        return encodedJobOffers.index_select(0, offerIndices);
    }
}


std::pair<Eigen::ArrayXd, Eigen::ArrayXd> DecisionNetHandler::choose_offers(
    const torch::Tensor& offerIndices,
    const Eigen::ArrayXd& prodFuncParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto encodedOffers = getEncodedOffersForOfferCreation(offerIndices);
    auto netOutput = offerNet->forward(
        encodedOffers,
        eigenToTorch(prodFuncParams),
        torch::tensor({money}),
        torch::tensor({labor}),
        eigenToTorch(inventory)
    );

    auto amounts_params = netOutput.index({"...", torch::tensor({0, 1})});
    auto amount_pair = sample_sigmoidNormal(amounts_params);
    auto amounts = torchToEigen(amount_pair.first) * inventory;

    auto prices_params = netOutput.index({"...", torch::tensor({2, 3})});
    auto price_pair = sample_logNormal(prices_params);
    auto prices = torchToEigen(price_pair.first);

    offerNetLogProba[time] += (
        torch::sum(amount_pair.second)
        + torch::sum(price_pair.second)
    );

    return std::make_pair(amounts, prices);
}


std::pair<double, double> DecisionNetHandler::choose_job_offers(
    const torch::Tensor& offerIndices,
    const Eigen::ArrayXd& prodFuncParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto encodedOffers = getEncodedOffersForJobOfferCreation(offerIndices);
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

    jobOfferNetLogProba[time] += (labor_pair.second + wage_pair.second)[0];

    return std::make_pair(totalLabor, wage);
}

} // namespace neural