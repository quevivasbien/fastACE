#include "neuralPersonDecisionMaker.h"


NeuralPersonDecisionMaker::NeuralPersonDecisionMaker(
    std::shared_ptr<UtilMaxer> parent,
    std::shared_ptr<NeuralDecisionMaker> guide
) : parent(parent), guide(guide) {}

NeuralPersonDecisionMaker::NeuralPersonDecisionMaker() : parent(nullptr), guide(nullptr) {}


void NeuralPersonDecisionMaker::check_guide_is_current() {
    if (parent->get_time() > guide->time) {
        guide->time_step();
    }
}

Eigen::ArrayXd get_utilParams() const {
    // NOTE: This only works if the parent has a CES utility function
    auto utilFunc = std::static_pointer_cast<CES>(parent->get_utilFunc());
    Eigen::ArrayXd utilParams(utilFunc->get_numInputs() + 2);
    utilParams << utilFunc->tfp, utilFunc->shareParams, utilFunc->substitutionParam;
    return utilParams;
}


std::vector<Order<Offer>> NeuralPersonDecisionMaker::choose_goods() {
    check_guide_is_current();

    // randomly select offers from encoded offers in guide->encodedOffers
    auto offerIndices = torch::randint(
        guide->purchaseNet->stackSize, guide->numEncodedOffers, torch::dtype(torch::kInt)
    );

    // get & return offer requests
    return guide->get_offers_to_request(
        offerIndices,
        get_utilParams(),
        parent->get_money(),
        parent->get_laborSupplied(),
        parent->get_inventory()
    );
}


std::vector<Order<JobOffer>> NeuralPersonDecisionMaker::choose_jobs() {
    check_guide_is_current();

    // randomly select job offers from guide->encodedJobOffers
    auto offerIndices = torch::randint(
        guide->laborSearchNet->stackSize, guide->numEncodedOffers, torch::dtype(torch::kInt)
    );

    // get & return offer requests
    return guide->get_joboffers_to_request(
        offerIndices,
        get_utilParams(),
        parent->get_money(),
        parent->get_laborSupplied(),
        parent->get_inventory()
    );

}


Eigen::ArrayXd NeuralPersonDecisionMaker::choose_goods_to_consume() {
    // do something...
}
