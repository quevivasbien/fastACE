#include <algorithm>
#include <random>
#include <limits>
#include "utilMaxer.h"

auto rd = std::random_device();
auto rng = std::default_random_engine(rd());


UtilMaxer::UtilMaxer(
    Economy* economy
) : Person(economy), utilFunc(std::make_shared<CobbDouglas>()) {}

UtilMaxer::UtilMaxer(
    Economy* economy,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToScalar> utilFunc
) : Person(economy, inventory, money), utilFunc(utilFunc) {}

double UtilMaxer::u(const Eigen::ArrayXd& quantities) {
    return utilFunc->f(quantities);
}

std::vector<std::shared_ptr<const Offer>> filterAvailable(
    const std::vector<std::shared_ptr<const Offer>>& offers,
    bool shuffle = true
) {
    std::vector<std::shared_ptr<const Offer>> availOffers;
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
    std::vector<unsigned int> numTaken,
    const Eigen::ArrayXd& quantities
) {
    double base_u = u(quantities);
    double best_util_per_cost = 0.0;
    int bestIdx = -1;
    for (int i = 0; i < numOffers; i++) {
        if ((offers[i]->amount_left > numTaken[i]) && (offers[i]->price <= budgetLeft)) {
            double du_per_cost = (
                (u(quantities + offers[i]->quantities) - base_u) / offers[i]->price;
            )
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
    std::vector<unsigned int>& numTaken,
    Eigen::ArrayXd& quantities
) {
    int bestIdx = 0;
    while (bestIdx != -1) {
        // repeats until nothing is affordable or gives positive util diff
        bestIdx = find_best_offer(
            availOffers, numOffers, budgetLeft, numTaken, quantities
        );
        quantities += availOffers[bestIdx]->quantities;
        budgetLeft -= availOffers[bestIdx]->price;
        numTaken[bestIdx]++;
    }
}

int UtilMaxer::find_worst_offer(
    const std::vector<std::shared_ptr<const Offer>>& offers,
    unsigned int numOffers,
    std::vector<unsigned int> numTaken,
    const Eigen::ArrayXd& quantities
) {
    double base_u = u(quantities);
    double worst_util_per_cost = std::numeric_limits<double>::infinity();
    int worstIdx = 0;
    for (int i = 0; i < numOffers; i++) {
        if (numTaken[i] > 0) {
            double du_per_cost = (
                (base_u - u(quantities - offers[i]->quantities)) / offers[i]->price;
            )
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
    std::vector<unsigned int>& numTaken,
    Eigen::ArrayXd& quantities,
    int heat
) {
    for (unsigned int i = 0; i < heat; i++) {
        int worstIdx = find_worst_offer(
            availOffers, numOffers, numTaken, quantities
        );
        quantities -= availOffers[worstIdx]->quantities;
        budgetLeft += availOffers[worstIdx]->price;
        numTaken[bestIdx]--;
    }
}

std::vector<Order> UtilMaxer::choose_goods(
    double budget,
    const std::vector<std::shared_ptr<const Offer>>& offers,
    int heat = 5,
    bool shuffle = true
) {
    // set to optimize over is not continuous, so we can't use standard approach
    // thus this is a nasty version of a knapsack problem
    // can't use linear programming either, since util is not linear
    // the algorithm used here gets only an approximate solution in most cases

    const std::vector<std::shared_ptr<const Offer>> availOffers = filterAvailable(offers, shuffle);
    unsigned int numOffers = availOffers.size();
    std::vector<unsigned int> numTaken(numOffers);  // number of each offer taken

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
    
    std::vector<Order> orders;
    for (unsigned int i = 0; i < numOffers; i++) {
        if (numTaken[i] > 0) {
            orders.push_back(Order(availOffers[i], numTaken[i]);
        }
    }
    return orders;
}

