#include "neuralFirmDecisionMaker.h"

namespace neural {

// firms will create their offers in discrete increments of this quantity
const double AMOUNT_PER_OFFER = 1.0;
const double LABOR_AMOUNT_PER_OFFER = 0.5;


NeuralFirmDecisionMaker::NeuralFirmDecisionMaker(
    std::shared_ptr<ProfitMaxer> parent,
    std::shared_ptr<DecisionNetHandler> guide
) : FirmDecisionMaker(parent), guide(guide) {}

NeuralFirmDecisionMaker::NeuralFirmDecisionMaker(
    std::shared_ptr<DecisionNetHandler> guide
) : NeuralFirmDecisionMaker(nullptr, guide) {}


void NeuralFirmDecisionMaker::check_guide_is_current() {
    if (parent->get_time() > guide->time) {
        guide->time_step();
    }
}

Eigen::ArrayXd NeuralFirmDecisionMaker::get_prodFuncParams() const {
    // NOTE: This only works if the parent has a SumOfVecToVec production function
    // with VToVFromVToS<CES> innerFunctions.
    auto prodFunc = std::static_pointer_cast<const SumOfVecToVec>(parent->get_prodFunc());
    Eigen::ArrayXd prodFuncParams((prodFunc->numInputs + 2) * prodFunc->numInnerFunctions);
    for (unsigned int i = 0; i < prodFunc->numInnerFunctions; i++) {
        auto ces = std::static_pointer_cast<const VToVFromVToS<CES>>(prodFunc->innerFunctions[i])->vecToScalar;
        prodFuncParams << ces->tfp, ces->shareParams, ces->substitutionParam;
    }
    return prodFuncParams;
}


std::vector<Order<Offer>> NeuralFirmDecisionMaker::choose_goods() {
    check_guide_is_current();

    // get & return offer requests
    return guide->firm_get_offers_to_request(
        get_prodFuncParams(),
        parent->get_money(),
        parent->get_laborHired(),
        parent->get_inventory()
    );
}


Eigen::ArrayXd NeuralFirmDecisionMaker::choose_production_inputs() {
    check_guide_is_current();

    return parent->get_inventory() * guide->get_production_proportions(
        get_prodFuncParams(),
        parent->get_money(),
        parent->get_laborHired(),
        parent->get_inventory()
    );
}


std::vector<std::shared_ptr<Offer>> NeuralFirmDecisionMaker::choose_good_offers() {
    check_guide_is_current();

    auto pair = guide->choose_offers(
        get_prodFuncParams(),
        parent->get_money(),
        parent->get_laborHired(),
        parent->get_inventory()
    );

    Eigen::ArrayXd amounts = pair.first;
    Eigen::ArrayXd prices = pair.second;

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
    check_guide_is_current();

    auto pair = guide->choose_job_offers(
        get_prodFuncParams(),
        parent->get_money(),
        parent->get_laborHired(),
        parent->get_inventory()
    );

    double laborAmount = pair.first;
    double wage = pair.second;

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
        return std::vector<std::shared_ptr<JobOffer>>();
    }
}

} // namespace neural
