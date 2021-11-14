#include "neuralPersonDecisionMaker.h"

namespace neural {

NeuralPersonDecisionMaker::NeuralPersonDecisionMaker(
    std::shared_ptr<UtilMaxer> parent,
    std::shared_ptr<DecisionNetHandler> guide
) : PersonDecisionMaker(parent), guide(guide) {}

NeuralPersonDecisionMaker::NeuralPersonDecisionMaker(
    std::shared_ptr<DecisionNetHandler> guide
) : NeuralPersonDecisionMaker(nullptr, guide) {}


void NeuralPersonDecisionMaker::check_guide_is_current() {
    if (parent->get_time() > guide->time) {
        guide->time_step();
    }
}

Eigen::ArrayXd NeuralPersonDecisionMaker::get_utilParams() const {
    // NOTE: This only works if the parent has a CES utility function
    auto utilFunc = std::static_pointer_cast<const CES>(parent->get_utilFunc());
    Eigen::ArrayXd utilParams(utilFunc->numInputs + 2);
    utilParams << utilFunc->tfp, utilFunc->shareParams, utilFunc->substitutionParam;
    return utilParams;
}


std::vector<Order<Offer>> NeuralPersonDecisionMaker::choose_goods() {
    check_guide_is_current();

    // get & return offer requests
    return guide->get_offers_to_request(
        get_utilParams(),
        parent->get_money(),
        parent->get_laborSupplied(),
        parent->get_inventory()
    );
}


std::vector<Order<JobOffer>> NeuralPersonDecisionMaker::choose_jobs() {
    check_guide_is_current();

    // get & return offer requests
    return guide->get_joboffers_to_request(
        get_utilParams(),
        parent->get_money(),
        parent->get_laborSupplied(),
        parent->get_inventory()
    );

}


Eigen::ArrayXd NeuralPersonDecisionMaker::choose_goods_to_consume() {
    check_guide_is_current();

    return parent->get_inventory() * guide->get_consumption_proportions(
        get_utilParams(),
        parent->get_money(),
        parent->get_laborSupplied(),
        parent->get_inventory()
    );
}

} // namespace neural
