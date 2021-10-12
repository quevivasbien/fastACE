#include <algorithm>
#include <random>
#include <limits>
#include "utilMaxer.h"

auto rng = std::default_random_engine {};


UtilMaxer::UtilMaxer(
    Economy* economy
) : Person(economy), utilFunc(std::make_shared<CobbDouglas>(economy->get_numGoods())) {}

UtilMaxer::UtilMaxer(
    Economy* economy,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToScalar> utilFunc
) : Person(economy, inventory, money), utilFunc(utilFunc) {}

double UtilMaxer::u(const Eigen::ArrayXd& quantities) {
    return utilFunc->f(quantities);
}

void UtilMaxer::buy_goods() {
    std::vector<Order<Offer>> orders = choose_goods(money, economy->get_market());
    for (auto order : orders) {
        for (unsigned int i; i < order.amount; i++) {
            respond_to_offer(order.offer);
            // TODO: Handle cases where response is rejected
        }
    }
}

template <typename T>
std::vector<std::shared_ptr<const T>> filter_available(
    const std::vector<std::shared_ptr<const T>>& offers,
    bool shuffle
) {
    std::vector<std::shared_ptr<const T>> availOffers;
    availOffers.reserve(offers.size());
    for (auto offer : offers) {
        if (offer->is_available()) {
            availOffers.push_back(offer);
        }
    }
    if (shuffle) {
        std::shuffle(std::begin(availOffers), std::end(availOffers), rng);
    }
    return availOffers;
}

int UtilMaxer::find_best_offer(
    const std::vector<std::shared_ptr<const Offer>>& offers,
    unsigned int numOffers,
    double budgetLeft,
    const Eigen::ArrayXi& numTaken,
    const Eigen::ArrayXd& quantities
) {
    double base_u = u(quantities);
    double best_util_per_cost = 0.0;
    int bestIdx = -1;
    for (int i = 0; i < numOffers; i++) {
        if ((offers[i]->amountLeft > numTaken(i)) && (offers[i]->price <= budgetLeft)) {
            double du_per_cost = (
                (u(quantities + offers[i]->quantities) - base_u) / offers[i]->price
            );
            if (du_per_cost > best_util_per_cost) {
                best_util_per_cost = du_per_cost;
                bestIdx = i;
            }
        }
    }
    return bestIdx;
}

void UtilMaxer::fill_basket(
    const std::vector<std::shared_ptr<const Offer>>& availOffers,
    unsigned int numOffers,
    double& budgetLeft,
    Eigen::ArrayXi& numTaken,
    Eigen::ArrayXd& quantities
) {
    int bestIdx = find_best_offer(
        availOffers, numOffers, budgetLeft, numTaken, quantities
    );
    while (bestIdx != -1) {
        // repeats until nothing is affordable or gives positive util diff
        quantities += availOffers[bestIdx]->quantities;
        budgetLeft -= availOffers[bestIdx]->price;
        numTaken(bestIdx)++;
        bestIdx = find_best_offer(
            availOffers, numOffers, budgetLeft, numTaken, quantities
        );
    }
}

int UtilMaxer::find_worst_offer(
    const std::vector<std::shared_ptr<const Offer>>& offers,
    unsigned int numOffers,
    const Eigen::ArrayXi& numTaken,
    const Eigen::ArrayXd& quantities
) {
    double base_u = u(quantities);
    double worst_util_per_cost = std::numeric_limits<double>::infinity();
    int worstIdx = 0;
    for (int i = 0; i < numOffers; i++) {
        if (numTaken(i) > 0) {
            double du_per_cost = (
                (base_u - u(quantities - offers[i]->quantities)) / offers[i]->price
            );
            if (du_per_cost < worst_util_per_cost) {
                worst_util_per_cost = du_per_cost;
                worstIdx = i;
            }
        }
    }
    return worstIdx;
}

void UtilMaxer::empty_basket(
    const std::vector<std::shared_ptr<const Offer>>& availOffers,
    unsigned int numOffers,
    double& budgetLeft,
    Eigen::ArrayXi& numTaken,
    Eigen::ArrayXd& quantities,
    int heat
) {
    for (unsigned int i = 0; i < heat; i++) {
        int worstIdx = find_worst_offer(
            availOffers, numOffers, numTaken, quantities
        );
        quantities -= availOffers[worstIdx]->quantities;
        budgetLeft += availOffers[worstIdx]->price;
        numTaken(worstIdx)--;
    }
}

std::vector<Order<Offer>> UtilMaxer::choose_goods(
    double budget,
    const std::vector<std::shared_ptr<const Offer>>& offers
) {
    return choose_goods(budget, offers, 5, true);
}

std::vector<Order<Offer>> UtilMaxer::choose_goods(
    double budget,
    const std::vector<std::shared_ptr<const Offer>>& offers,
    int heat,
    bool shuffle
) {
    // set to optimize over is not continuous, so we can't use standard approach
    // thus this is a nasty version of a knapsack problem
    // can't use linear programming either, since util is not linear
    // the algorithm used here gets only an approximate solution in most cases

    const std::vector<std::shared_ptr<const Offer>> availOffers = filter_available<Offer>(offers, shuffle);
    unsigned int numOffers = availOffers.size();
    Eigen::ArrayXi numTaken = Eigen::ArrayXi::Zero(numOffers);  // number of each offer taken

    double budgetLeft = budget;
    Eigen::ArrayXd quantities = Eigen::ArrayXd::Zero(economy->get_numGoods());

    fill_basket(
        availOffers, numOffers, budgetLeft, numTaken, quantities
    );
    int offersInBasket = numTaken.sum();
    heat = ((heat <= offersInBasket) ? heat : offersInBasket) - 1;
    while (heat > 0) {
        empty_basket(
            availOffers, numOffers, budgetLeft, numTaken, quantities, heat
        );
        fill_basket(
            availOffers, numOffers, budgetLeft, numTaken, quantities
        );
        offersInBasket = numTaken.sum();
        heat = ((heat <= offersInBasket) ? heat : offersInBasket) - 1;
    }

    std::vector<Order<Offer>> orders;
    for (unsigned int i = 0; i < numOffers; i++) {
        if (numTaken(i) > 0) {
            orders.push_back(Order<Offer>(availOffers[i], numTaken(i)));
        }
    }
    return orders;
}

void UtilMaxer::consume_goods() {
    // just consumes all goods
    inventory = Eigen::ArrayXd::Zero(inventory.size());
}


void UtilMaxer::search_for_jobs() {
    std::vector<Order<JobOffer>> orders = choose_jobs(labor, economy->get_jobMarket());
    for (auto order : orders) {
        for (unsigned int i; i < order.amount; i++) {
            respond_to_jobOffer(order.offer);
            // TODO: Handle cases where response is rejected
        }
    }
}

int UtilMaxer::find_best_jobOffer(
    const std::vector<std::shared_ptr<const JobOffer>>& offers,
    unsigned int numOffers,
    double laborLeft,
    const Eigen::ArrayXi& numTaken
) {
    double best_wage_per_labor = 0.0;
    int bestIdx = -1;
    for (int i = 0; i < numOffers; i++) {
        if ((offers[i]->amountLeft > numTaken(i)) && (offers[i]->labor <= laborLeft)) {
            double wage_per_labor = offers[i]->wage / offers[i]->labor;
            if (wage_per_labor > best_wage_per_labor) {
                best_wage_per_labor = wage_per_labor;
                bestIdx = i;
            }
        }
    }
    return bestIdx;
}

void UtilMaxer::fill_labor_basket(
    const std::vector<std::shared_ptr<const JobOffer>>& availOffers,
    unsigned int numOffers,
    double& laborLeft,
    Eigen::ArrayXi& numTaken
) {
    int bestIdx = find_best_jobOffer(
        availOffers, numOffers, laborLeft, numTaken
    );
    while (bestIdx != -1) {
        // repeats until nothing is affordable or gives positive util diff
        laborLeft -= availOffers[bestIdx]->labor;
        numTaken(bestIdx)++;
        bestIdx = find_best_jobOffer(
            availOffers, numOffers, laborLeft, numTaken
        );
    }
}

int UtilMaxer::find_worst_jobOffer(
    const std::vector<std::shared_ptr<const JobOffer>>& offers,
    unsigned int numOffers,
    const Eigen::ArrayXi& numTaken
) {
    double worst_wage_per_labor = std::numeric_limits<double>::infinity();
    int worstIdx = 0;
    for (int i = 0; i < numOffers; i++) {
        if (numTaken(i) > 0) {
            double wage_per_labor = offers[i]->wage / offers[i]->labor;
            if (wage_per_labor < worst_wage_per_labor) {
                worst_wage_per_labor = wage_per_labor;
                worstIdx = i;
            }
        }
    }
    return worstIdx;
}

void UtilMaxer::empty_labor_basket(
    const std::vector<std::shared_ptr<const JobOffer>>& availOffers,
    unsigned int numOffers,
    double& laborLeft,
    Eigen::ArrayXi& numTaken,
    int heat
) {
    for (unsigned int i = 0; i < heat; i++) {
        int worstIdx = find_worst_jobOffer(
            availOffers, numOffers, numTaken
        );
        laborLeft += availOffers[worstIdx]->labor;
        numTaken(worstIdx)--;
    }
}

std::vector<Order<JobOffer>> UtilMaxer::choose_jobs(
    double laborBudget,
    const std::vector<std::shared_ptr<const JobOffer>>& jobOffers
) {
    return choose_jobs(laborBudget, jobOffers, 5, true);
}


std::vector<Order<JobOffer>> UtilMaxer::choose_jobs(
    double laborBudget,
    const std::vector<std::shared_ptr<const JobOffer>>& jobOffers,
    int heat,
    bool shuffle
) {
    // TODO: heat analogy is not quite right for this problem, so modify implementation
    const std::vector<std::shared_ptr<const JobOffer>> availOffers = filter_available<JobOffer>(jobOffers, shuffle);
    unsigned int numOffers = availOffers.size();
    Eigen::ArrayXi numTaken = Eigen::ArrayXi::Zero(numOffers);  // number of each offer taken

    double laborLeft = laborBudget;

    fill_labor_basket(
        availOffers, numOffers, laborLeft, numTaken
    );
    int offersInBasket = numTaken.sum();
    heat = ((heat <= offersInBasket) ? heat : offersInBasket) - 1;
    while (heat > 0) {
        empty_labor_basket(
            availOffers, numOffers, laborLeft, numTaken, heat
        );
        fill_labor_basket(
            availOffers, numOffers, laborLeft, numTaken
        );
        offersInBasket = numTaken.sum();
        heat = ((heat <= offersInBasket) ? heat : offersInBasket) - 1;
    }

    std::vector<Order<JobOffer>> orders;
    for (unsigned int i = 0; i < numOffers; i++) {
        if (numTaken(i) > 0) {
            orders.push_back(Order<JobOffer>(availOffers[i], numTaken(i)));
        }
    }
    return orders;
}
