#include "neuralEconomy.h"

namespace neural {

NeuralPersonDecisionMaker::NeuralPersonDecisionMaker(
    std::weak_ptr<UtilMaxer> parent,
    std::weak_ptr<DecisionNetHandler> guide
) : PersonDecisionMaker(parent), guide(guide) {}

NeuralPersonDecisionMaker::NeuralPersonDecisionMaker(
    std::weak_ptr<DecisionNetHandler> guide
) : guide(guide) {}


void NeuralPersonDecisionMaker::confirm_synchronized() {
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);

    guide_->synchronize_time(parent_);
    if (parent_->get_time() > time) {
        utilParams = get_utilParams();
        myOfferIndices = guide_->generate_offerIndices();
        myJobOfferIndices = guide_->generate_jobOfferIndices();
        record_state_value();
        time++;
    }
}

Eigen::ArrayXd NeuralPersonDecisionMaker::get_utilParams() const {
    auto parent_ = parent.lock();
    assert(parent_ != nullptr);
    // NOTE: This only works if the parent has a CES utility function
    auto utilFunc = std::static_pointer_cast<const CES>(parent_->get_utilFunc());
    Eigen::ArrayXd utilParams(utilFunc->numInputs + 2);
    utilParams << utilFunc->tfp, utilFunc->shareParams, utilFunc->substitutionParam;
    return utilParams;
}

void NeuralPersonDecisionMaker::record_state_value() {
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);
    guide_->record_value(
        parent_.get(),
        myOfferIndices,
        myJobOfferIndices,
        utilParams,
        parent_->get_money(),
        parent_->get_laborSupplied(),
        parent_->get_inventory()
    );
}


std::vector<Order<Offer>> NeuralPersonDecisionMaker::choose_goods() {
    confirm_synchronized();
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);

    // get & return offer requests
    return guide_->get_offers_to_request(
        parent_.get(),
        myOfferIndices,
        utilParams,
        parent_->get_money(),
        parent_->get_laborSupplied(),
        parent_->get_inventory()
    );
}


std::vector<Order<JobOffer>> NeuralPersonDecisionMaker::choose_jobs() {
    confirm_synchronized();
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);

    // get & return offer requests
    return guide_->get_joboffers_to_request(
        parent_.get(),
        myJobOfferIndices,
        utilParams,
        parent_->get_money(),
        parent_->get_laborSupplied(),
        parent_->get_inventory()
    );

}


Eigen::ArrayXd NeuralPersonDecisionMaker::choose_goods_to_consume() {
    confirm_synchronized();
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);

    Eigen::ArrayXd to_consume = parent_->get_inventory() * guide_->get_consumption_proportions(
        parent_.get(),
        utilParams,
        parent_->get_money(),
        parent_->get_laborSupplied(),
        parent_->get_inventory()
    );

    double util = parent_->u(to_consume);
    guide_->record_reward(parent_.get(), util);

    return to_consume;
}

} // namespace neural
