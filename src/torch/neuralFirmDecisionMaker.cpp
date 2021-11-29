#include "neuralEconomy.h"

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


void NeuralFirmDecisionMaker::confirm_synchronized() {
    if (parent->get_time() > guide->time) {
        guide->time_step();
    }
    if (parent->get_time() > time) {
        prodFuncParams = get_prodFuncParams();
        myOfferIndices = guide->firm_generate_offerIndices();
        myJobOfferIndices = guide->firm_generate_jobOfferIndices();
        record_state_value();
        record_profit();
        time++;
    }
}

Eigen::ArrayXd NeuralFirmDecisionMaker::get_prodFuncParams() const {
    // NOTE: This only works if the parent has a SumOfVecToVec production function
    // with VToVFromVToS<CES> innerFunctions.
    auto prodFunc = std::static_pointer_cast<const SumOfVecToVec>(parent->get_prodFunc());
    unsigned int componentSize = prodFunc->numInputs + 2;
    Eigen::ArrayXXd prodFuncParams(componentSize, prodFunc->numInnerFunctions);
    for (unsigned int i = 0; i < prodFunc->numInnerFunctions; i++) {
        auto ces = std::static_pointer_cast<const VToVFromVToS<CES>>(prodFunc->innerFunctions[i])->vecToScalar;
        prodFuncParams.col(i) << ces->tfp, ces->shareParams, ces->substitutionParam;
    }
    return Eigen::Map<Eigen::ArrayXd>(prodFuncParams.data(), prodFuncParams.size());
}

void NeuralFirmDecisionMaker::record_state_value() {
    guide->firm_record_value(
        parent,
        myOfferIndices,
        myJobOfferIndices,
        prodFuncParams,
        parent->get_money(),
        parent->get_laborHired(),
        parent->get_inventory()
    );
}

void NeuralFirmDecisionMaker::record_profit() {
    if (time > 0) {
        double profit = parent->get_money() - last_money;
        guide->record_reward(parent, profit, 1);
    }
    last_money = parent->get_money();
}


std::vector<Order<Offer>> NeuralFirmDecisionMaker::choose_goods() {
    confirm_synchronized();
    
    // get & return offer requests
    return guide->firm_get_offers_to_request(
        parent,
        myOfferIndices,
        prodFuncParams,
        parent->get_money(),
        parent->get_laborHired(),
        parent->get_inventory()
    );
}


Eigen::ArrayXd NeuralFirmDecisionMaker::choose_production_inputs() {
    confirm_synchronized();

    return parent->get_inventory() * guide->get_production_proportions(
        parent,
        prodFuncParams,
        parent->get_money(),
        parent->get_laborHired(),
        parent->get_inventory()
    );
}


std::vector<std::shared_ptr<Offer>> NeuralFirmDecisionMaker::choose_good_offers() {
    confirm_synchronized();

    auto amt_price_pair = guide->choose_offers(
        parent,
        myOfferIndices,
        prodFuncParams,
        parent->get_money(),
        parent->get_laborHired(),
        parent->get_inventory()
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

    auto labor_wage_pair = guide->choose_job_offers(
        parent,
        myJobOfferIndices,
        prodFuncParams,
        parent->get_money(),
        parent->get_laborHired(),
        parent->get_inventory()
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
        return std::vector<std::shared_ptr<JobOffer>>();
    }
}

} // namespace neural
