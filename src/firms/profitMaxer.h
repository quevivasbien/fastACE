#ifndef PROFITMAXER_H
#define PROFITMAXER_H

#include "base.h"
#include "vecToVec.h"

class ProfitMaxer;

struct GoodProducer {
    GoodProducer(ProfitMaxer* parent);
    virtual Eigen::ArrayXd choose_production_inputs() = 0;
    ProfitMaxer* parent;
};

struct GoodSeller {
    GoodSeller(ProfitMaxer* parent);
    virtual std::vector<std::shared_ptr<Offer>> choose_good_offers() = 0;
    ProfitMaxer* parent;
};

struct LaborFinder {
    LaborFinder(ProfitMaxer* parent);
    virtual std::vector<std::shared_ptr<JobOffer>> choose_job_offers() = 0;
    ProfitMaxer* parent;
};

class ProfitMaxer : public Firm {
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> create(Args&& ... args);

    Eigen::ArrayXd f(const Eigen::ArrayXd& quantities);
protected:
    ProfitMaxer(
        Economy* economy,
        std::shared_ptr<Agent> owner,
        unsigned int outputIndex
    );
    ProfitMaxer(
        Economy* economy,
        std::shared_ptr<Agent> owner,
        std::shared_ptr<VecToVec> prodFunc,
        std::shared_ptr<GoodProducer> goodProducer,
        std::shared_ptr<GoodSeller> goodSeller,
        std::shared_ptr<LaborFinder> laborFinder
    );
    ProfitMaxer(
        Economy* economy,
        std::vector<std::shared_ptr<Agent>> owners,
        Eigen::ArrayXd inventory,
        double money,
        std::shared_ptr<VecToVec> prodFunc,
        std::shared_ptr<GoodProducer> goodProducer,
        std::shared_ptr<GoodSeller> goodSeller,
        std::shared_ptr<LaborFinder> laborFinder
    );

    // calls goodProducer to figure out what inventory to use to produce
    virtual void produce() override;
    virtual void sell_goods() override;
    virtual void search_for_laborers() override;

    // prodFunc should have economy->numGoods + 1 inputs and economy->numGoods outputs
    // extra input is labor, which is always the first input
    std::shared_ptr<VecToVec> prodFunc;
    std::shared_ptr<GoodProducer> goodProducer;
    std::shared_ptr<GoodSeller> goodSeller;
    std::shared_ptr<LaborFinder> laborFinder;
};

struct GreedyGoodProducer : GoodProducer {
    GreedyGoodProducer(ProfitMaxer* parent);
    virtual Eigen::ArrayXd choose_production_inputs() override;
};

struct GreedyGoodSeller : GoodSeller {
    GreedyGoodSeller(ProfitMaxer* parent);
    // uses all goods in inventory for production
    virtual std::vector<std::shared_ptr<Offer>> choose_good_offers() override;
};

struct GreedyLaborFinder : LaborFinder {
    GreedyLaborFinder(ProfitMaxer* parent);
    virtual std::vector<std::shared_ptr<JobOffer>> choose_job_offers() override;
};

#endif
