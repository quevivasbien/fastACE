#include <limits>
#include "utilMaxer.h"

PersonDecisionMaker::PersonDecisionMaker() {}

PersonDecisionMaker::PersonDecisionMaker(std::weak_ptr<UtilMaxer> parent) : parent(parent) {}


UtilMaxer::UtilMaxer(
    std::shared_ptr<Economy> economy,
    std::shared_ptr<VecToScalar> utilFunc,
    double discountRate,
    std::shared_ptr<PersonDecisionMaker> decisionMaker
) : Person(economy),
    utilFunc(utilFunc),
    discountRate(discountRate),
    decisionMaker(decisionMaker)
{}

UtilMaxer::UtilMaxer(
    std::shared_ptr<Economy> economy,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToScalar> utilFunc,
    double discountRate,
    std::shared_ptr<PersonDecisionMaker> decisionMaker
) : Person(economy, inventory, money),
    utilFunc(utilFunc),
    discountRate(discountRate),
    decisionMaker(decisionMaker)
{}

void UtilMaxer::init_decisionMaker() {
    assert(decisionMaker->parent.lock() == nullptr);
    decisionMaker->parent = std::static_pointer_cast<UtilMaxer>(shared_from_this());
}

std::shared_ptr<const VecToScalar> UtilMaxer::get_utilFunc() const {
    return utilFunc;
}

std::shared_ptr<const PersonDecisionMaker> UtilMaxer::get_decisionMaker() const {
    return decisionMaker;
}

double UtilMaxer::get_discountRate() const {
    return discountRate;
}

std::string UtilMaxer::get_typename() const {
    return "UtilMaxer";
}

double UtilMaxer::u(double labor, const Eigen::ArrayXd& quantities) {
    Eigen::ArrayXd inputs(utilFunc->numInputs);
    inputs << 1 - labor, quantities;
    return utilFunc->f(inputs);
}

double UtilMaxer::u(const Eigen::ArrayXd& quantities) {
    return u(laborSupplied, quantities);
}

void UtilMaxer::buy_goods() {
    std::vector<Order<Offer>> orders = decisionMaker->choose_goods();
    for (auto order : orders) {
        for (unsigned int i = 0; i < order.amount; i++) {
            respond_to_offer(order.offer);
            // TODO: Handle cases where response is rejected
        }
    }
}


void UtilMaxer::search_for_jobs() {
    std::vector<Order<JobOffer>> orders = decisionMaker->choose_jobs();
    for (auto order : orders) {
        for (unsigned int i = 0; i < order.amount; i++) {
            respond_to_jobOffer(order.offer);
            // TODO: Handle cases where response is rejected
        }
    }
}


void UtilMaxer::consume_goods() {
    // just consumes all goods
    std::lock_guard<std::mutex> lock(myMutex);
    inventory -= decisionMaker->choose_goods_to_consume();
}
