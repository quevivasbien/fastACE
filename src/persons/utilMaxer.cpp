#include <limits>
#include "utilMaxer.h"

PersonDecisionMaker::PersonDecisionMaker(std::shared_ptr<UtilMaxer> parent) : parent(parent) {}

UtilMaxer::UtilMaxer(
    Economy* economy
) : Person(economy),
    utilFunc(std::make_shared<CobbDouglas>(economy->get_numGoods())),
    decisionMaker(std::make_shared<BasicPersonDecisionMaker>())
{}

UtilMaxer::UtilMaxer(
    Economy* economy,
    std::shared_ptr<VecToScalar> utilFunc,
    std::shared_ptr<PersonDecisionMaker> decisionMaker
) : Person(economy),
    utilFunc(utilFunc),
    decisionMaker(decisionMaker)
{}

UtilMaxer::UtilMaxer(
    Economy* economy,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToScalar> utilFunc,
    std::shared_ptr<PersonDecisionMaker> decisionMaker
) : Person(economy, inventory, money),
    utilFunc(utilFunc),
    decisionMaker(decisionMaker)
{}

void UtilMaxer::init_decisionMaker() {
    assert(decisionMaker->parent == nullptr);
    decisionMaker->parent = std::static_pointer_cast<UtilMaxer>(shared_from_this());
}

double UtilMaxer::u(const Eigen::ArrayXd& quantities) {
    return utilFunc->f(quantities);
}

void UtilMaxer::buy_goods() {
    std::vector<Order<Offer>> orders = decisionMaker->choose_goods();
    for (auto order : orders) {
        for (unsigned int i; i < order.amount; i++) {
            respond_to_offer(order.offer);
            // TODO: Handle cases where response is rejected
        }
    }
}


void UtilMaxer::search_for_jobs() {
    std::vector<Order<JobOffer>> orders = decisionMaker->choose_jobs();
    for (auto order : orders) {
        for (unsigned int i; i < order.amount; i++) {
            respond_to_jobOffer(order.offer);
            // TODO: Handle cases where response is rejected
        }
    }
}


void UtilMaxer::consume_goods() {
    // just consumes all goods
    inventory -= decisionMaker->choose_goods_to_consume();
}


namespace BasicPersonDecisionMakerHelperClasses {

struct GoodChooser {
    GoodChooser(
        std::shared_ptr<UtilMaxer> parent,
        double money,
        const std::vector<std::shared_ptr<const Offer>>& offers,
        unsigned int numOffers,
        unsigned int numGoods
    ) : parent(parent),
        offers(offers),
        numOffers(numOffers),
        budgetLeft(money),
        numTaken(Eigen::ArrayXi::Zero(numOffers)),
        quantities(Eigen::ArrayXd::Zero(numGoods))
    {}

    std::shared_ptr<UtilMaxer> parent;
    std::vector<std::shared_ptr<const Offer>> offers;
    unsigned int numOffers;
    double budgetLeft;
    Eigen::ArrayXi numTaken;
    Eigen::ArrayXd quantities;
    int heat = constants::heat;

    int find_best_offer() {
        double base_u = parent->u(quantities);
        double best_util_per_cost = 0.0;
        int bestIdx = -1;
        for (int i = 0; i < numOffers; i++) {
            if ((offers[i]->amountLeft > numTaken(i)) && (offers[i]->price <= budgetLeft)) {
                double du_per_cost = (
                    (parent->u(quantities + offers[i]->quantities) - base_u) / offers[i]->price
                );
                if (du_per_cost > best_util_per_cost) {
                    best_util_per_cost = du_per_cost;
                    bestIdx = i;
                }
            }
        }
        return bestIdx;
    }

    void fill_basket() {
        int bestIdx = find_best_offer();
        while (bestIdx != -1) {
            // repeats until nothing is affordable or gives positive util diff
            quantities += offers[bestIdx]->quantities;
            budgetLeft -= offers[bestIdx]->price;
            numTaken(bestIdx)++;
            bestIdx = find_best_offer();
        }
    }

    int find_worst_offer() {
        double base_u = parent->u(quantities);
        double worst_util_per_cost = std::numeric_limits<double>::infinity();
        int worstIdx = 0;
        for (int i = 0; i < numOffers; i++) {
            if (numTaken(i) > 0) {
                double du_per_cost = (
                    (base_u - parent->u(quantities - offers[i]->quantities)) / offers[i]->price
                );
                if (du_per_cost < worst_util_per_cost) {
                    worst_util_per_cost = du_per_cost;
                    worstIdx = i;
                }
            }
        }
        return worstIdx;
    }

    void empty_basket() {
        for (unsigned int i = 0; i < heat; i++) {
            int worstIdx = find_worst_offer();
            quantities -= offers[worstIdx]->quantities;
            budgetLeft += offers[worstIdx]->price;
            numTaken(worstIdx)--;
        }
    }

    std::vector<Order<Offer>> choose_goods() {
        fill_basket();
        int offersInBasket = numTaken.sum();
        int heat = ((heat <= offersInBasket) ? heat : offersInBasket) - 1;
        while (heat > 0) {
            empty_basket();
            fill_basket();
            offersInBasket = numTaken.sum();
            heat = ((heat <= offersInBasket) ? heat : offersInBasket) - 1;
        }

        std::vector<Order<Offer>> orders;
        for (unsigned int i = 0; i < numOffers; i++) {
            if (numTaken(i) > 0) {
                orders.push_back(Order<Offer>(offers[i], numTaken(i)));
            }
        }
        return orders;
    }
};


struct JobChooser {
    JobChooser(
        const std::vector<std::shared_ptr<const JobOffer>>& offers,
        unsigned int numOffers,
        double labor
    ) : offers(offers),
        numOffers(numOffers),
        laborLeft(labor),
        numTaken(Eigen::ArrayXi::Zero(numOffers))
    {}

    std::vector<std::shared_ptr<const JobOffer>> offers;
    unsigned int numOffers;
    double laborLeft;
    Eigen::ArrayXi numTaken;


    int find_best_jobOffer() {
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

    void fill_labor_basket() {
        int bestIdx = find_best_jobOffer();
        while (bestIdx != -1) {
            // repeats until nothing is affordable or gives positive util diff
            laborLeft -= offers[bestIdx]->labor;
            numTaken(bestIdx)++;
            bestIdx = find_best_jobOffer();
        }
    }

    std::vector<Order<JobOffer>> choose_jobs() {
        fill_labor_basket();
        std::vector<Order<JobOffer>> orders;
        for (unsigned int i = 0; i < numOffers; i++) {
            if (numTaken(i) > 0) {
                orders.push_back(Order<JobOffer>(offers[i], numTaken(i)));
            }
        }
        return orders;
    }

};

}


BasicPersonDecisionMaker::BasicPersonDecisionMaker() : BasicPersonDecisionMaker(nullptr) {}
BasicPersonDecisionMaker::BasicPersonDecisionMaker(std::shared_ptr<UtilMaxer> parent) : PersonDecisionMaker(parent) {}

std::vector<Order<Offer>> BasicPersonDecisionMaker::choose_goods() {
    // set to optimize over is not continuous, so we can't use standard approach
    // thus this is a nasty version of a knapsack problem
    // can't use linear programming either, since util is not linear
    // the algorithm used here gets only an approximate solution in most cases

    const auto availOffers = filter_available<Offer>(
        parent->get_economy()->get_market(),
        parent->get_economy()->get_rng()
    );
    BasicPersonDecisionMakerHelperClasses::GoodChooser goodChooser(
        parent,
        parent->get_money(),
        availOffers,
        availOffers.size(),
        parent->get_economy()->get_numGoods()
    );
    return goodChooser.choose_goods();
}

std::vector<Order<JobOffer>> BasicPersonDecisionMaker::choose_jobs() {
    const auto availOffers = filter_available<JobOffer>(
        parent->get_economy()->get_jobMarket(),
        parent->get_economy()->get_rng()
    );
    BasicPersonDecisionMakerHelperClasses::JobChooser jobChooser(
        availOffers,
        availOffers.size(),
        parent->get_laborSupplied()
    );
    return jobChooser.choose_jobs();
}

Eigen::ArrayXd BasicPersonDecisionMaker::choose_goods_to_consume() {
    return parent->get_inventory();
}
