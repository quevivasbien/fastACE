#include "neuralPersonDecisionMaker.h"

namespace neural {

NeuralPersonDecisionMaker::NeuralPersonDecisionMaker(
    std::shared_ptr<UtilMaxer> parent,
    std::shared_ptr<DecisionNetHandler> guide
) : PersonDecisionMaker(parent), guide(guide) {}

NeuralPersonDecisionMaker::NeuralPersonDecisionMaker(
    std::shared_ptr<DecisionNetHandler> guide
) : NeuralPersonDecisionMaker(nullptr, guide) {}


void NeuralPersonDecisionMaker::confirm_synchronized() {
    guide->synchronize_time(parent);
    if (parent->get_time() > time) {
        utilParams = get_utilParams();
        myOfferIndices = guide->generate_offerIndices();
        myJobOfferIndices = guide->generate_jobOfferIndices();
        record_state_value();
        time++;
    }
}

Eigen::ArrayXd NeuralPersonDecisionMaker::get_utilParams() const {
    // NOTE: This only works if the parent has a CES utility function
    auto utilFunc = std::static_pointer_cast<const CES>(parent->get_utilFunc());
    Eigen::ArrayXd utilParams(utilFunc->numInputs + 2);
    utilParams << utilFunc->tfp, utilFunc->shareParams, utilFunc->substitutionParam;
    return utilParams;
}

void NeuralPersonDecisionMaker::record_state_value() {
    guide->record_value(
        parent,
        myOfferIndices,
        myJobOfferIndices,
        utilParams,
        parent->get_money(),
        parent->get_laborSupplied(),
        parent->get_inventory()
    );
}


std::vector<Order<Offer>> NeuralPersonDecisionMaker::choose_goods() {
    confirm_synchronized();

    // get & return offer requests
    return guide->get_offers_to_request(
        parent,
        myOfferIndices,
        utilParams,
        parent->get_money(),
        parent->get_laborSupplied(),
        parent->get_inventory()
    );
}


std::vector<Order<JobOffer>> NeuralPersonDecisionMaker::choose_jobs() {
    confirm_synchronized();

    // get & return offer requests
    return guide->get_joboffers_to_request(
        parent,
        myJobOfferIndices,
        utilParams,
        parent->get_money(),
        parent->get_laborSupplied(),
        parent->get_inventory()
    );

}


Eigen::ArrayXd NeuralPersonDecisionMaker::choose_goods_to_consume() {
    confirm_synchronized();

    Eigen::ArrayXd to_consume = parent->get_inventory() * guide->get_consumption_proportions(
        parent,
        utilParams,
        parent->get_money(),
        parent->get_laborSupplied(),
        parent->get_inventory()
    );

    double util = parent->u(to_consume);
    guide->record_reward(parent, util);

    return to_consume;
}

} // namespace neural
