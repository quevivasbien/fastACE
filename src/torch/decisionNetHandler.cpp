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
    update_encodedOffers();
    update_encodedJobOffers();
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

    update_encodedOffers();
    update_encodedJobOffers();
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


void DecisionNetHandler::time_step() {
    std::lock_guard<std::mutex> lock(myMutex);
    update_encodedOffers();
    update_encodedJobOffers();
    time++;
}


std::vector<Order<Offer>> DecisionNetHandler::create_offer_requests(
    torch::Tensor offerIndices, // dtype = kInt64
    torch::Tensor purchase_probas
) {
    auto to_purchase = (torch::rand(purchase_probas.sizes()) < purchase_probas);

    std::vector<Order<Offer>> toRequest;
    for (int i = 0; i < to_purchase.size(0); i++) {
        if (to_purchase[i].item<bool>()) {
            auto offer = offers[offerIndices[i].item<int>()];
            toRequest.push_back(Order<Offer>(offer, 1));
        }
    }
    return toRequest;
}


std::vector<Order<Offer>> DecisionNetHandler::get_offers_to_request(
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    if (numEncodedOffers == 0) {
        // if no offers available, return empty list
        return std::vector<Order<Offer>>();
    }
    // get random indices of offers to consider
    auto offerIndices = torch::randint(
        0, numEncodedOffers, purchaseNet->stackSize, torch::dtype(torch::kInt64)
    );
    auto probas = get_purchase_probas(
        offerIndices, utilParams, budget, labor, inventory, purchaseNet, encodedOffers
    );

    return create_offer_requests(offerIndices, probas);
}


std::vector<Order<Offer>> DecisionNetHandler::firm_get_offers_to_request(
    const Eigen::ArrayXd& prodFuncParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    if (numEncodedOffers == 0) {
        // if no offers available, return empty list
        return std::vector<Order<Offer>>();
    }
    // get random indices of offers to consider
    auto offerIndices = torch::randint(
        0, numEncodedOffers, firmPurchaseNet->stackSize, torch::dtype(torch::kInt64)
    );
    auto probas = get_purchase_probas(
        offerIndices, prodFuncParams, budget, labor, inventory, firmPurchaseNet, encodedOffers
    );

    return create_offer_requests(offerIndices, probas);
}


std::vector<Order<JobOffer>> DecisionNetHandler::create_joboffer_requests(
    torch::Tensor offerIndices, // dtype = kInt64
    torch::Tensor job_probas
) {
    auto to_take = (torch::rand(job_probas.sizes()) < job_probas);

    std::vector<Order<JobOffer>> toRequest;
    for (int i = 0; i < to_take.size(0); i++) {
        if (to_take[i].item<bool>()) {
            auto jobOffer = jobOffers[offerIndices[i].item<int>()];
            toRequest.push_back(Order<JobOffer>(jobOffer, 1));
        }
    }
    return toRequest;
}


std::vector<Order<JobOffer>> DecisionNetHandler::get_joboffers_to_request(
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    if (numEncodedJobOffers == 0) {
        // if no offers available, return empty list
        return std::vector<Order<JobOffer>>();
    }
    // get random indices of offers to consider
    auto offerIndices = torch::randint(
        0, numEncodedJobOffers, laborSearchNet->stackSize, torch::dtype(torch::kInt64)
    );
    auto probas = get_job_probas(
        offerIndices, utilParams, money, labor, inventory, laborSearchNet, encodedJobOffers
    );

    return create_joboffer_requests(offerIndices, probas);
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
    return torchToEigen(production_pair.first);
}


torch::Tensor DecisionNetHandler::getEncodedOffersForOfferCreation() {
    if (numEncodedJobOffers == 0) {
        // return a tensor of zeros of the appropriate shape, meaning no offers currently on market
        return torch::zeros({offerNet->stackSize, offerEncoder->encodingSize});
    }
    else {
        auto offerIndices = torch::randint(
            0, numEncodedJobOffers, offerNet->stackSize, torch::dtype(torch::kInt64)
        );
        return encodedOffers.index_select(0, offerIndices);
    }
}


std::pair<Eigen::ArrayXd, Eigen::ArrayXd> DecisionNetHandler::choose_offers(
    const Eigen::ArrayXd& prodFuncParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto encodedOffers = getEncodedOffersForOfferCreation();
    auto netOutput = offerNet->forward(
        encodedOffers,
        eigenToTorch(prodFuncParams),
        torch::tensor({money}),
        torch::tensor({labor}),
        eigenToTorch(inventory)
    );
    auto amounts_params = netOutput.index({"...", torch::tensor({0, 1})});
    auto amounts = torchToEigen(
        sample_sigmoidNormal(amounts_params).first
    ) * inventory;
    auto prices_params = netOutput.index({"...", torch::tensor({2, 3})});
    auto prices = torchToEigen(
        sample_logNormal(prices_params).first
    );

    return std::make_pair(amounts, prices);
}


std::pair<double, double> DecisionNetHandler::choose_job_offers(
    const Eigen::ArrayXd& prodFuncParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto encodedOffers = getEncodedOffersForOfferCreation();
    auto netOutput = jobOfferNet->forward(
        encodedOffers,
        eigenToTorch(prodFuncParams),
        torch::tensor({money}),
        torch::tensor({labor}),
        eigenToTorch(inventory)
    );
    auto labor_params = netOutput.index({"...", torch::tensor({0, 1})});
    double totalLabor = sample_logNormal(labor_params).first.item<double>();
    auto wage_params = netOutput.index({"...", torch::tensor({2, 3})});
    double wage = sample_logNormal(wage_params).first.item<double>();

    return std::make_pair(totalLabor, wage);
}

} // namespace neural
