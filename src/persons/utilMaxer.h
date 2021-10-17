#ifndef UTILMAXER_H
#define UTILMAXER_H

#include "base.h"
#include "vecToScalar.h"
// #include "solve.h"


template <typename T>
std::vector<std::shared_ptr<const T>> filter_available(
    const std::vector<std::shared_ptr<const T>>& offers,
    bool shuffle
);


class UtilMaxer;

struct GoodChooser {
    GoodChooser(UtilMaxer* parent);
    // called by UtilMaxer::buy_goods()
    // looks at current goods on market and chooses bundle that maximizes utility
    // subject to restriction that total price is within budget
    virtual std::vector<Order<Offer>> choose_goods() = 0;
    UtilMaxer* parent;
};

struct JobChooser {
    JobChooser(UtilMaxer* parent);
    // analogous to GoodChooser::choose_goods(), but for jobs
    // in default implementation does not take utility of labor into account
    // i.e. only tries to maximize wages
    virtual std::vector<Order<JobOffer>> choose_jobs() = 0;
    UtilMaxer* parent;
};

struct GoodConsumer {
    GoodConsumer(UtilMaxer* parent);
    virtual Eigen::ArrayXd choose_goods_to_consume() = 0;
    UtilMaxer* parent;
};


class UtilMaxer : public Person {
public:
    template <typename T, typename ... Args>
    friend std::shared_ptr<T> create(Args&& ... args);

    double u(const Eigen::ArrayXd& quantities);  // alias for utilFunc.f
protected:
    UtilMaxer(Economy* economy);
    UtilMaxer(
        Economy* economy,
        std::shared_ptr<VecToScalar> utilFunc,
        std::shared_ptr<GoodChooser> goodChooser,
        std::shared_ptr<JobChooser> jobChooser,
        std::shared_ptr<GoodConsumer> goodConsumer
    );
    UtilMaxer(
        Economy* economy,
        Eigen::ArrayXd inventory,
        double money,
        std::shared_ptr<VecToScalar> utilFunc,
        std::shared_ptr<GoodChooser> goodChooser,
        std::shared_ptr<JobChooser> jobChooser,
        std::shared_ptr<GoodConsumer> goodConsumer
    );

    std::shared_ptr<VecToScalar> utilFunc;
    std::shared_ptr<GoodChooser> goodChooser;
    std::shared_ptr<JobChooser> jobChooser;
    std::shared_ptr<GoodConsumer> goodConsumer;

    // calls gooChooser->choose_goods(), then requests those goods
    virtual void buy_goods() override;

    // calls jobChooser->choose_jobs(), then requests those jobs
    virtual void search_for_jobs() override;

    // calls goodConsumer->choose_goods_to_consume(), then removes those goods from inventory
    virtual void consume_goods() override;
};

struct GreedyGoodChooser : GoodChooser {
    GreedyGoodChooser(UtilMaxer* parent);

    // helper functions for choose_goods():
    int find_best_offer(
        const std::vector<std::shared_ptr<const Offer>>& offers,
        unsigned int numOffers,
        double budgetLeft,
        const Eigen::ArrayXi& numTaken,
        const Eigen::ArrayXd& quantities
    );

    void fill_basket(
        const std::vector<std::shared_ptr<const Offer>>& availOffers,
        unsigned int numOffers,
        double& budgetLeft,
        Eigen::ArrayXi& numTaken,
        Eigen::ArrayXd& quantities
    );

    int find_worst_offer(
        const std::vector<std::shared_ptr<const Offer>>& offers,
        unsigned int numOffers,
        const Eigen::ArrayXi& numTaken,
        const Eigen::ArrayXd& quantities
    );

    void empty_basket(
        const std::vector<std::shared_ptr<const Offer>>& availOffers,
        unsigned int numOffers,
        double& budgetLeft,
        Eigen::ArrayXi& numTaken,
        Eigen::ArrayXd& quantities,
        int heat
    );

    std::vector<Order<Offer>> choose_goods() override;
    virtual std::vector<Order<Offer>> choose_goods(
        // analogy to simulated annealing, more heat means more iterations for optimization
        int heat,
        // whether to shuffle offers before going over them, default is true
        bool shuffle
    );
};

struct GreedyJobChooser : JobChooser {
    GreedyJobChooser(UtilMaxer* parent);

    int find_best_jobOffer(
        const std::vector<std::shared_ptr<const JobOffer>>& offers,
        unsigned int numOffers,
        double laborLeft,
        const Eigen::ArrayXi& numTaken
    );

    void fill_labor_basket(
        const std::vector<std::shared_ptr<const JobOffer>>& availOffers,
        unsigned int numOffers,
        double& laborLeft,
        Eigen::ArrayXi& numTaken
    );

    int find_worst_jobOffer(
        const std::vector<std::shared_ptr<const JobOffer>>& offers,
        unsigned int numOffers,
        const Eigen::ArrayXi& numTaken
    );

    void empty_labor_basket(
        const std::vector<std::shared_ptr<const JobOffer>>& availOffers,
        unsigned int numOffers,
        double& laborLeft,
        Eigen::ArrayXi& numTaken,
        int heat
    );

    virtual std::vector<Order<JobOffer>> choose_jobs() override;
    virtual std::vector<Order<JobOffer>> choose_jobs(int heat, bool shuffle);
};

struct GreedyGoodConsumer : GoodConsumer {
    GreedyGoodConsumer(UtilMaxer* parent);

    virtual Eigen::ArrayXd choose_goods_to_consume() override;
};

#endif
