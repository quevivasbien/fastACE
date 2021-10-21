#include <limits>
#include "profitMaxer.h"


FirmDecisionMaker::FirmDecisionMaker(std::shared_ptr<ProfitMaxer> parent) : parent(parent) {}


ProfitMaxer::ProfitMaxer(Economy* economy, std::shared_ptr<Agent> owner, unsigned int outputIndex) :
    Firm(economy, owner),
    prodFunc(
        std::make_shared<VToVFromVToS>(
            std::make_shared<CobbDouglas>(economy->get_numGoods() + 1),
            economy->get_numGoods(),
            outputIndex
        )
    ),
    decisionMaker(
        std::make_shared<BasicFirmDecisionMaker>()
    )
{}

ProfitMaxer::ProfitMaxer(
    Economy* economy,
    std::shared_ptr<Agent> owner,
    std::shared_ptr<VecToVec> prodFunc,
    std::shared_ptr<FirmDecisionMaker> decisionMaker
) : Firm(economy, owner),
    prodFunc(prodFunc),
    decisionMaker(decisionMaker)
{
    assert(prodFunc->get_numInputs() == economy->get_numGoods() + 1);
}

ProfitMaxer::ProfitMaxer(
    Economy* economy,
    std::vector<std::shared_ptr<Agent>> owners,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToVec> prodFunc,
    std::shared_ptr<FirmDecisionMaker> decisionMaker
) : Firm(economy, owners, inventory, money),
    prodFunc(prodFunc),
    decisionMaker(decisionMaker)
{
    assert(prodFunc->get_numInputs() == economy->get_numGoods() + 1);
}

void ProfitMaxer::init_decisionMaker() {
    // this assertion is to make sure that the decisionMaker doesn't get assigned to more than one ProfitMaxer
    assert(decisionMaker->parent == nullptr);
    decisionMaker->parent = std::static_pointer_cast<ProfitMaxer>(shared_from_this());
}


Eigen::ArrayXd ProfitMaxer::f(double labor, const Eigen::ArrayXd& quantities) {
    Eigen::ArrayXd inputs(prodFunc->get_numInputs());
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


void ProfitMaxer::produce() {
    Eigen::ArrayXd inputs = decisionMaker->choose_production_inputs();
    inventory += (f(laborHired, inputs) - inputs);
}

void ProfitMaxer::sell_goods() {
    auto offers = decisionMaker->choose_good_offers();
    for (auto offer : offers) {
        post_offer(offer);
    }
}

void ProfitMaxer::search_for_laborers() {
    auto jobOffers = decisionMaker->choose_job_offers();
    for (auto offer : jobOffers) {
        post_jobOffer(offer);
    }
}

void ProfitMaxer::buy_goods() {
    auto orders = decisionMaker->choose_goods();
    for (auto order : orders) {
        for (unsigned int i; i < order.amount; i++) {
            respond_to_offer(order.offer);
            // TODO: Handle cases where response is rejected
        }
    }
}


BasicFirmDecisionMaker::BasicFirmDecisionMaker(std::shared_ptr<ProfitMaxer> parent) : FirmDecisionMaker(parent) {}
BasicFirmDecisionMaker::BasicFirmDecisionMaker() : BasicFirmDecisionMaker(nullptr) {}

Eigen::ArrayXd BasicFirmDecisionMaker::choose_production_inputs() {
    // just uses all available goods
    return parent->get_inventory();
}

double BasicFirmDecisionMaker::choose_price(unsigned int idx) {
    double price = constants::defaultPrice;
    for (auto offer : lastOffers) {
        if (offer->quantities(idx) > 0) {
            if (offer->amountTaken > 0) {
                price = offer->price * constants::priceMultiplier;
            }
            else {
                price = offer->price / constants::priceMultiplier;
            }
            break;
        }
    }
    return price;
}

void BasicFirmDecisionMaker::create_offer(
    unsigned int idx,
    std::vector<std::shared_ptr<Offer>>& offers,
    double amountToOffer,
    unsigned int numGoods
) {
    // look for offers for this good in lastOffers
    // use this to set the price for this good
    double price = choose_price(idx);
    Eigen::ArrayXd quantities = Eigen::ArrayXd::Zero(numGoods);
    quantities(idx) = 1.0;
    offers.push_back(std::make_shared<Offer>(parent, amountToOffer, quantities, price));
}

std::vector<std::shared_ptr<Offer>> BasicFirmDecisionMaker::choose_good_offers() {
    // the basic idea is to look through offers made last round
    // if those offers sold, increase the price
    // if they did not sell, reduce the price
    // if there are no matching offers from last round, start out at a default price
    // everything in inventory will be listed for sale
    unsigned int numGoods = parent->get_economy()->get_numGoods();
    Eigen::ArrayXd inventory = parent->get_inventory();
    std::vector<std::shared_ptr<Offer>> offers;
    for (unsigned int i = 0; i < numGoods; i++) {
        if (inventory(i) > 0) {
            create_offer(i, offers, inventory(i), numGoods);
        }
    }
    lastOffers = offers;
    return offers;
}

namespace BasicFirmDecisionMakerHelperClasses {
struct GoodChooser {

    GoodChooser(
        std::shared_ptr<ProfitMaxer> parent,
        double labor,
        double money,
        const std::vector<std::shared_ptr<const Offer>>& offers,
        const Eigen::ArrayXd& sellingPrices
    ) : parent(parent),
        labor(labor),
        budgetLeft(money),
        offers(offers),
        sellingPrices(sellingPrices),
        numOffers(offers.size()),
        quantities(Eigen::ArrayXd::Zero(sellingPrices.size())),
        numTaken(Eigen::ArrayXi::Zero(sellingPrices.size()))
    {}

    std::shared_ptr<ProfitMaxer> parent;
    std::vector<std::shared_ptr<const Offer>> offers;
    unsigned int numOffers;
    double budgetLeft;
    Eigen::ArrayXd quantities;
    Eigen::ArrayXi numTaken;
    double labor;
    Eigen::ArrayXd sellingPrices;
    int heat = constants::heat;


    int find_best_offer() {
        double base_rev = parent->get_revenue(labor, quantities, sellingPrices);
        double best_rev_per_cost = 0.0;
        int bestIdx = -1;
        for (int i = 0; i < numOffers; i++) {
            if ((offers[i]->amountLeft > numTaken(i)) && (offers[i]->price <= budgetLeft)) {
                double rev_per_cost = (
                    (parent->get_revenue(labor, quantities + offers[i]->quantities, sellingPrices) - base_rev)
                        / offers[i]->price
                );
                if (rev_per_cost > best_rev_per_cost) {
                    best_rev_per_cost = rev_per_cost;
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
        double base_rev = parent->get_revenue(labor, quantities, sellingPrices);
        double worst_rev_per_cost = std::numeric_limits<double>::infinity();
        int worstIdx = 0;
        for (int i = 0; i < numOffers; i++) {
            if (numTaken(i) > 0) {
                double rev_per_cost = (
                    (base_rev - parent->get_revenue(
                        labor, quantities - offers[i]->quantities, sellingPrices
                    )) / offers[i]->price
                );
                if (rev_per_cost < worst_rev_per_cost) {
                    worst_rev_per_cost = rev_per_cost;
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

    std::vector<Order<Offer>> get_orders() {
        fill_basket();
        int offersInBasket = numTaken.sum();
        heat = ((heat <= offersInBasket) ? heat : offersInBasket) - 1;
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
}


Eigen::ArrayXd BasicFirmDecisionMaker::get_total_quantities(const std::vector<Order<Offer>>& orders) {
    Eigen::ArrayXd sum = Eigen::ArrayXd::Zero(parent->get_economy()->get_numGoods());
    for (auto order : orders) {
        sum += (order.offer->quantities * order.amount);
    }
    return sum;
}

void BasicFirmDecisionMaker::compare_budgets(
    double newBudget,
    double& bestBudget,
    double& bestRevenue,
    const std::vector<std::shared_ptr<const Offer>> offers,
    const Eigen::ArrayXd& sellingPrices
) {
    double newLabor = (parent->get_laborHired() + 0.01) * newBudget / laborBudget;
    double goodsBudget = parent->get_money() * (1 - newBudget) / (1 - laborBudget);
    double newRevenue = parent->get_revenue(
        newLabor,
        get_total_quantities(choose_goods(newLabor, goodsBudget, offers, sellingPrices)),
        sellingPrices
    );
    if (newRevenue > bestRevenue) {
        bestBudget = newBudget;
        bestRevenue = newRevenue;
    }
}

void BasicFirmDecisionMaker::evaluate_labor_demand(
    const std::vector<Order<Offer>>& defaultOrders,
    const std::vector<std::shared_ptr<const Offer>> offers,
    const Eigen::ArrayXd& sellingPrices
) {
    // try out higher and lower labor budgets to figure out whether to change labor budget
    double bestBudget = laborBudget;
    double bestRevenue = parent->get_revenue(
        parent->get_laborHired(),
        get_total_quantities(defaultOrders),
        sellingPrices
    );
    if (laborBudget < 1.0) {
        double newBudget = laborBudget * constants::priceMultiplier;
        if (newBudget > 1.0) {
            newBudget = 1.0;
        }
        compare_budgets(newBudget, bestBudget, bestRevenue, offers, sellingPrices);
    }
    compare_budgets(laborBudget / constants::priceMultiplier, bestBudget, bestRevenue, offers, sellingPrices);
    laborBudget = bestBudget;
}

std::vector<Order<Offer>> BasicFirmDecisionMaker::choose_goods() {
    // get approximate prices at which goods can be sold
    unsigned int numGoods = parent->get_economy()->get_numGoods();
    Eigen::ArrayXd sellingPrices = Eigen::ArrayXd::Zero(numGoods);
    for (unsigned int i = 0; i < numGoods; i++) {
        sellingPrices(i) = choose_price(i);
    }
    // get available offers
    auto availOffers = filter_available<Offer>(parent->get_economy()->get_market(), parent->get_economy()->get_rng());
    // now just plug into big boy friend with default values
    return choose_goods(parent->get_laborHired(), parent->get_money(), availOffers, sellingPrices);
}

std::vector<Order<Offer>> BasicFirmDecisionMaker::choose_goods(
    double labor,
    double money,
    const std::vector<std::shared_ptr<const Offer>>& availOffers,
    const Eigen::ArrayXd& sellingPrices
) {
    // go through a process similar to the one done by UtilMaxer::choose_goods()
    BasicFirmDecisionMakerHelperClasses::GoodChooser goodChooser(parent, labor, money, availOffers, sellingPrices);
    auto orders = goodChooser.get_orders();

    if (labor == parent->get_laborHired()) {
        evaluate_labor_demand(orders, availOffers, sellingPrices);
    }

    return orders;
}

std::vector<std::shared_ptr<JobOffer>> BasicFirmDecisionMaker::choose_job_offers() {
    // works similarly to choose_good_offers()
    for (auto jobOffer : lastJobOffers) {
        if (jobOffer->amountTaken > 0) {
            wage *= constants::priceMultiplier;
        }
        else {
            wage /= constants::priceMultiplier;
        }
        jobOffer->amountLeft = 0;
    }
    unsigned int numToOffer = parent->get_money() * laborBudget / constants::laborIncrement;
    std::vector<std::shared_ptr<JobOffer>> jobOffers = {
        std::make_shared<JobOffer>(parent, numToOffer, constants::laborIncrement, wage)
    };
    return jobOffers;
}
