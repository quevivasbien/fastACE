#include <limits>
#include "profitMaxer.h"

FirmDecisionMaker::FirmDecisionMaker() {}

FirmDecisionMaker::FirmDecisionMaker(std::weak_ptr<ProfitMaxer> parent) : parent(parent) {}


ProfitMaxer::ProfitMaxer(
    std::shared_ptr<Economy> economy,
    std::shared_ptr<Agent> owner,
    std::shared_ptr<VecToVec> prodFunc,
    std::shared_ptr<FirmDecisionMaker> decisionMaker
) : Firm(economy, owner),
    prodFunc(prodFunc),
    decisionMaker(decisionMaker)
{
    assert(prodFunc->numInputs == economy->get_numGoods() + 1);
}

ProfitMaxer::ProfitMaxer(
    std::shared_ptr<Economy> economy,
    std::vector<std::shared_ptr<Agent>> owners,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToVec> prodFunc,
    std::shared_ptr<FirmDecisionMaker> decisionMaker
) : Firm(economy, owners, inventory, money),
    prodFunc(prodFunc),
    decisionMaker(decisionMaker)
{
    assert(prodFunc->numInputs == economy->get_numGoods() + 1);
}

void ProfitMaxer::init_decisionMaker() {
    // this assertion is to make sure that the decisionMaker doesn't get assigned to more than one ProfitMaxer
    assert(decisionMaker->parent.lock() == nullptr);
    decisionMaker->parent = std::static_pointer_cast<ProfitMaxer>(shared_from_this());
}

std::string ProfitMaxer::get_typename() const {
    return "ProfitMaxer";
}


Eigen::ArrayXd ProfitMaxer::f(double labor, const Eigen::ArrayXd& quantities) {
    Eigen::ArrayXd inputs(prodFunc->numInputs);
    inputs << labor, quantities;
    return prodFunc->f(inputs);
}

double ProfitMaxer::get_revenue(
    double labor,
    const Eigen::ArrayXd& quantities,
    const Eigen::ArrayXd& prices
) {
    return f(labor, quantities).matrix().dot(prices.matrix());
}

std::shared_ptr<const VecToVec> ProfitMaxer::get_prodFunc() const {
    return prodFunc;
}

std::shared_ptr<const FirmDecisionMaker> ProfitMaxer::get_decisionMaker() const {
    return decisionMaker;
}


void ProfitMaxer::produce() {
    std::lock_guard<std::mutex> lock(myMutex);
    Eigen::ArrayXd inputs = decisionMaker->choose_production_inputs();
    inventory += (f(laborHired, inputs) - inputs);
}

void ProfitMaxer::sell_goods() {
    auto newOffers = decisionMaker->choose_good_offers();
    // remove last round's offers from the market before posting new offers
    {
        std::lock_guard<std::mutex> lock(myMutex);
        for (auto offer : myOffers) {
            offer->amountLeft = 0;
        }
    }
    for (auto offer : newOffers) {
        post_offer(offer);
    }
}

void ProfitMaxer::search_for_laborers() {
    auto newJobOffers = decisionMaker->choose_job_offers();
    {
        std::lock_guard<std::mutex> lock(myMutex);
        // remove last round's offers from the market before posting new offers
        for (auto offer : myJobOffers) {
            offer->amountLeft = 0;
        }
    }
    for (auto offer : newJobOffers) {
        post_jobOffer(offer);
    }
}

void ProfitMaxer::buy_goods() {
    auto orders = decisionMaker->choose_goods();
    for (auto order : orders) {
        for (unsigned int i = 0; i < order.amount; i++) {
            respond_to_offer(order.offer);
            // TODO: Handle cases where response is rejected
        }
    }
}
