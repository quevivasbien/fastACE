#ifndef PROFITMAXER_H
#define PROFITMAXER_H

#include "base.h"
#include "constants.h"
#include "vecToVec.h"


class ProfitMaxer;

struct FirmDecisionMaker {
    virtual Eigen::ArrayXd choose_production_inputs() = 0;
    virtual std::vector<std::shared_ptr<Offer>> choose_good_offers() = 0;
    virtual std::vector<std::shared_ptr<JobOffer>> choose_job_offers() = 0;
    virtual std::vector<Order<Offer>> choose_goods() = 0;
    FirmDecisionMaker(std::shared_ptr<ProfitMaxer> parent);
    std::shared_ptr<ProfitMaxer> parent;
};


class ProfitMaxer : public Firm {
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> create(Args&& ... args);

    template <typename ... Args>
    static std::shared_ptr<ProfitMaxer> init(Args&& ... args) {
        auto profitMaxer = create<ProfitMaxer>(std::forward<Args>(args) ...);
        profitMaxer->init_decisionMaker();
        return profitMaxer;
    }

    Eigen::ArrayXd f(double labor, const Eigen::ArrayXd& quantities);
    double get_revenue(double labor, const Eigen::ArrayXd& quantities, const Eigen::ArrayXd& prices);

    std::shared_ptr<const VecToVec> get_prodFunc() const;

    virtual std::string get_typename() const override;

protected:
    ProfitMaxer(
        std::shared_ptr<Economy> economy,
        std::shared_ptr<Agent> owner,
        unsigned int outputIndex
    );
    ProfitMaxer(
        std::shared_ptr<Economy> economy,
        std::shared_ptr<Agent> owner,
        std::shared_ptr<VecToVec> prodFunc,
        std::shared_ptr<FirmDecisionMaker> decisionMaker
    );
    ProfitMaxer(
        std::shared_ptr<Economy> economy,
        std::vector<std::shared_ptr<Agent>> owners,
        Eigen::ArrayXd inventory,
        double money,
        std::shared_ptr<VecToVec> prodFunc,
        std::shared_ptr<FirmDecisionMaker> decisionMaker
    );

    void init_decisionMaker();

    // calls decisionMaker to determine behavior
    virtual void produce() override;
    virtual void sell_goods() override;
    virtual void search_for_laborers() override;
    virtual void buy_goods() override;

    // prodFunc should have economy->numGoods + 1 inputs and economy->numGoods outputs
    // extra input is labor, which is always the first input
    std::shared_ptr<VecToVec> prodFunc;
    std::shared_ptr<FirmDecisionMaker> decisionMaker;
};

struct BasicFirmDecisionMaker : FirmDecisionMaker {
public:
    BasicFirmDecisionMaker();
    BasicFirmDecisionMaker(std::shared_ptr<ProfitMaxer> parent);
    virtual Eigen::ArrayXd choose_production_inputs() override;
    double choose_price(unsigned int idx);
    void create_offer(
        unsigned int idx,
        std::vector<std::shared_ptr<Offer>>& offers,
        double amountToOffer,
        unsigned int numGoods
    );
    virtual std::vector<std::shared_ptr<Offer>> choose_good_offers() override;
    virtual std::vector<Order<Offer>> choose_goods() override;
    virtual std::vector<std::shared_ptr<JobOffer>> choose_job_offers() override;
protected:
    std::vector<std::shared_ptr<Offer>> lastOffers;
    std::vector<Order<Offer>> lastOrders;
    std::vector<std::shared_ptr<JobOffer>> lastJobOffers;
    double laborBudget = constants::defaultLaborBudget;
    double wage = constants::defaultWage;

    // helpers for choose_goods:
    Eigen::ArrayXd get_total_quantities(const std::vector<Order<Offer>>& orders);
    void compare_budgets(
        double newBudget,
        double& bestBudget,
        double& bestRevenue,
        const std::vector<std::shared_ptr<const Offer>> offers,
        const Eigen::ArrayXd& sellingPrices
    );
    void evaluate_labor_demand(
        const std::vector<Order<Offer>>& defaultOrders,
        const std::vector<std::shared_ptr<const Offer>> offers,
        const Eigen::ArrayXd& sellingPrices
    );
    std::vector<Order<Offer>> choose_goods(
        double labor,
        double money,
        const std::vector<std::shared_ptr<const Offer>>& availOffers,
        const Eigen::ArrayXd& sellingPrices
    );
};

#endif
