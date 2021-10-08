#include <algorithm>
#include <random>
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

const std::vector<std::shared_ptr<const Offer>> filterAvailable(
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

double find_cheapest(const std::vector<std::shared_ptr<const Offer>>& offers, unsigned int numOffers) {
    double cheapest_price = availOffers[0]->price;
    for (unsigned int i = 1; i < numOffers; i++) {
        if (availOffers[i]->price < cheapest_price) {
            cheapest_price = availOffers[i]->price;
        }
    }
    return cheapest_price;
}

void UtilMaxer::choose_goods(
    double budget,
    const std::vector<std::shared_ptr<const Offer>>& offers
) {
    // set to optimize over is not continuous, so we can't use standard approach
    // thus this is a nasty version of a knapsack problem
    // can't use linear programming either, since util is not linear
    // the algorithm used here gets only an approximate solution

    const std::vector<std::shared_ptr<const Offer>> availOffers = filterAvailable(offers);
    unsigned int numOffers = availOffers.size();
    std::vector<unsigned int> numTaken(numOffers);  // number of each offer taken

    double budget_left = budget;
    Eigen::ArrayXd quantities = Eigen::ArrayXd::Zero(economy->get_numGoods());

    // // find cheapest offer
    // double cheapest_price = find_cheapest(availOffers, numOffers);
    //
    // // first step is to randomly pick offers until budget is exhausted
    // unsigned int i = 0;
    // while (budget_left >= cheapest_price) {
    //     if ((availOffers[i]->price <= budget_left) && (availOffers[i]->amount_left > numTaken[i])) {
    //         numTaken[i]++;
    //         budget_left -= availOffers[i]->price;
    //         if ((availOffers[i]->amount_left = 0) && (availOffers[i]->price == cheapest_price)) {
    //
    //         }
    //         quantities += availOffers[i]->quantities;
    //     }
    //     i = (i + 1) % numOffers;
    // }
}
