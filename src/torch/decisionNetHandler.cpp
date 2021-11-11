#include "decisionNetHandler.h"

namespace neural {

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


torch::Tensor get_purchase_probas(
    const torch::Tensor& offerIndices, // dtype = kInt
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> purchaseNet,
    const torch::Tensor& encodedOffers
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
    const torch::Tensor& offerIndices, // dtype = kInt
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory,
    std::shared_ptr<PurchaseNet> laborSearchNet,
    const torch::Tensor& encodedJobOffers
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
    Economy* economy,
    std::shared_ptr<OfferEncoder> offerEncoder,
    std::shared_ptr<OfferEncoder> jobOfferEncoder,
    std::shared_ptr<PurchaseNet> purchaseNet,
    std::shared_ptr<PurchaseNet> firmPurchaseNet,
    std::shared_ptr<PurchaseNet> laborSearchNet,
    std::shared_ptr<ConsumptionNet> consumptionNet
) : economy(economy),
    offerEncoder(offerEncoder),
    jobOfferEncoder(jobOfferEncoder),
    purchaseNet(purchaseNet),
    firmPurchaseNet(firmPurchaseNet),
    laborSearchNet(laborSearchNet),
    consumptionNet(consumptionNet)
{
    update_encodedOffers();
    update_encodedJobOffers();
};


DecisionNetHandler::DecisionNetHandler(Economy* economy) : economy(economy) {
    int numAgents = economy->get_maxAgents();
    int numGoods = economy->get_numGoods();
    // NOTE: numUtilParams assumes persons have CES utility functions,
    // with goods + labor + tfp + elasticity as params
    // assumes firms have same number of params in their production functs
    int numUtilParams = numGoods + 3;

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
        numUtilParams,
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
    update_encodedOffers();
    update_encodedJobOffers();
    time++;
}


std::vector<Order<Offer>> DecisionNetHandler::create_offer_requests(
    const torch::Tensor& offerIndices, // dtype = kInt
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
    const torch::Tensor& offerIndices, // dtype = kInt
    const Eigen::ArrayXd& utilParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto probas = get_purchase_probas(
        offerIndices, utilParams, budget, labor, inventory, purchaseNet, encodedOffers
    );

    return create_offer_requests(offerIndices, probas);
}


std::vector<Order<Offer>> DecisionNetHandler::firm_get_offers_to_request(
    const torch::Tensor& offerIndices, // dtype = kInt
    const Eigen::ArrayXd& prodFuncParams,
    double budget,
    double labor,
    const Eigen::ArrayXd& inventory
) {
    auto probas = get_purchase_probas(
        offerIndices, prodFuncParams, budget, labor, inventory, firmPurchaseNet, encodedOffers
    );

    return create_offer_requests(offerIndices, probas);
}


std::vector<Order<JobOffer>> DecisionNetHandler::create_joboffer_requests(
    const torch::Tensor& offerIndices, // dtype = kInt
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
    const torch::Tensor& offerIndices, // dtype = kInt
    const Eigen::ArrayXd& utilParams,
    double money,
    double labor,
    const Eigen::ArrayXd& inventory
) {
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
    return torchToEigen(
        consumptionNet->forward(utilParams_, money_, labor_, inventory_)
    );
}

} // namespace neural
