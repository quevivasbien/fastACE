#include "neuralEconomy.h"

namespace neural {

// firms will create their offers in discrete increments of this quantity
const double AMOUNT_PER_OFFER = 1.0;
const double LABOR_AMOUNT_PER_OFFER = 0.5;


NeuralFirmDecisionMaker::NeuralFirmDecisionMaker(
    std::weak_ptr<ProfitMaxer> parent,
    std::weak_ptr<DecisionNetHandler> guide
) : FirmDecisionMaker(parent), guide(guide) {}

NeuralFirmDecisionMaker::NeuralFirmDecisionMaker(
    std::weak_ptr<DecisionNetHandler> guide
) : guide(guide) {}


void NeuralFirmDecisionMaker::confirm_synchronized() {
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);
    guide_->synchronize_time(parent_);
    if (parent_->get_time() > time) {
        prodFuncParams = get_prodFuncParams();
        myOfferIndices = guide_->firm_generate_offerIndices();
        myJobOfferIndices = guide_->firm_generate_jobOfferIndices();
        record_state_value();
        record_profit();
        time++;
    }
}

Eigen::ArrayXd NeuralFirmDecisionMaker::get_prodFuncParams() const {
    // NOTE: This only works if the parent has a SumOfVecToVec production function
    // with VToVFromVToS<CES> innerFunctions.
    auto parent_ = parent.lock();
    assert(parent_ != nullptr);
    auto prodFunc = std::static_pointer_cast<const SumOfVecToVec>(parent_->get_prodFunc());
    unsigned int componentSize = prodFunc->numInputs + 2;
    Eigen::ArrayXXd prodFuncParams(componentSize, prodFunc->numInnerFunctions);
    for (unsigned int i = 0; i < prodFunc->numInnerFunctions; i++) {
        auto ces = std::static_pointer_cast<const VToVFromVToS<CES>>(prodFunc->innerFunctions[i])->vecToScalar;
        prodFuncParams.col(i) << ces->tfp, ces->shareParams, ces->substitutionParam;
    }
    return Eigen::Map<Eigen::ArrayXd>(prodFuncParams.data(), prodFuncParams.size());
}

void NeuralFirmDecisionMaker::record_state_value() {
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);
    guide_->firm_record_value(
        parent_.get(),
        myOfferIndices,
        myJobOfferIndices,
        prodFuncParams,
        parent_->get_money(),
        parent_->get_laborHired(),
        parent_->get_inventory()
    );
}

void NeuralFirmDecisionMaker::record_profit() {
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);
    if (time > 0) {
        double profit = parent_->get_money() - last_money;
        guide_->record_reward(parent_, profit, 1);
    }
    last_money = parent_->get_money();
}


std::vector<Order<Offer>> NeuralFirmDecisionMaker::choose_goods() {
    confirm_synchronized();
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);

    // get & return offer requests
    return guide_->firm_get_offers_to_request(
        parent_.get(),
        myOfferIndices,
        prodFuncParams,
        parent_->get_money(),
        parent_->get_laborHired(),
        parent_->get_inventory()
    );
}


Eigen::ArrayXd NeuralFirmDecisionMaker::choose_production_inputs() {
    confirm_synchronized();
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);

    return parent_->get_inventory() * guide_->get_production_proportions(
        parent_.get(),
        prodFuncParams,
        parent_->get_money(),
        parent_->get_laborHired(),
        parent_->get_inventory()
    );
}


std::vector<std::shared_ptr<Offer>> NeuralFirmDecisionMaker::choose_good_offers() {
    confirm_synchronized();
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);

    auto amt_price_pair = guide_->choose_offers(
        parent_.get(),
        myOfferIndices,
        prodFuncParams,
        parent_->get_money(),
        parent_->get_laborHired(),
        parent_->get_inventory()
    );

    Eigen::ArrayXd amounts = amt_price_pair.first;
    Eigen::ArrayXd prices = amt_price_pair.second;

    Eigen::ArrayXi numOffers = (amounts / AMOUNT_PER_OFFER).cast<int>();

    int numGoods = amounts.size();
    std::vector<std::shared_ptr<Offer>> offers;
    for (int i = 0; i < numGoods; i++) {
        // make an Offer for each type of good
        if (numOffers(i) > 0) {
            Eigen::ArrayXd quantities = Eigen::ArrayXd::Zero(numGoods);
            quantities(i) = AMOUNT_PER_OFFER;
            offers.push_back(
                std::make_shared<Offer>(
                    parent, numOffers(i), quantities, prices(i) / AMOUNT_PER_OFFER
                )
            );
        }
    }
    return offers;
}


std::vector<std::shared_ptr<JobOffer>> NeuralFirmDecisionMaker::choose_job_offers() {
    confirm_synchronized();
    auto guide_ = guide.lock();
    auto parent_ = parent.lock();
    assert(guide_ != nullptr && parent_ != nullptr);

    auto labor_wage_pair = guide_->choose_job_offers(
        parent_.get(),
        myJobOfferIndices,
        prodFuncParams,
        parent_->get_money(),
        parent_->get_laborHired(),
        parent_->get_inventory()
    );

    double laborAmount = labor_wage_pair.first;
    double wage = labor_wage_pair.second;

    int numOffers = laborAmount / LABOR_AMOUNT_PER_OFFER;

    if (numOffers > 0) {
        std::vector<std::shared_ptr<JobOffer>> offers = {
            std::make_shared<JobOffer>(
                parent, numOffers, LABOR_AMOUNT_PER_OFFER, wage / LABOR_AMOUNT_PER_OFFER
            )
        };
        return offers;
    }
    else {
        return {};
    }
}

} // namespace neural
