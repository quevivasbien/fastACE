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
    // should leave parent unitialized at first
    // since ProfitMaxer::init will automatically assign decision maker to itself
    FirmDecisionMaker();
    std::weak_ptr<ProfitMaxer> parent;
    
protected:
    FirmDecisionMaker(std::weak_ptr<ProfitMaxer> parent);
};


class ProfitMaxer : public Firm {
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> util::create(Args&& ... args);

    template <typename ... Args>
    static std::shared_ptr<ProfitMaxer> init(Args&& ... args) {
        auto profitMaxer = util::create<ProfitMaxer>(std::forward<Args>(args) ...);
        profitMaxer->init_decisionMaker();
        return profitMaxer;
    }

    Eigen::ArrayXd f(double labor, const Eigen::ArrayXd& quantities);
    double get_revenue(double labor, const Eigen::ArrayXd& quantities, const Eigen::ArrayXd& prices);

    std::shared_ptr<const VecToVec> get_prodFunc() const;
    std::shared_ptr<const FirmDecisionMaker> get_decisionMaker() const;

    virtual std::string get_typename() const override;

protected:
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

#endif
