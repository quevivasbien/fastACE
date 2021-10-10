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


class UtilMaxer : public Person {
public:
    template <typename T, typename ... Args>
    friend std::shared_ptr<T> create(Args&& ... args);

    double u(const Eigen::ArrayXd& quantities);  // alias for utilFunc.f
protected:
    UtilMaxer(Economy* economy);
    UtilMaxer(
        Economy* economy,
        Eigen::ArrayXd inventory,
        double money,
        std::shared_ptr<VecToScalar> utilFunc
    );

    std::shared_ptr<VecToScalar> utilFunc;

    // calls choose_goods(), then requests those goods
    virtual void buy_goods() override;

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

    // called by buy_goods()
    // looks at current goods on market and chooses bundle that maximizes utility
    // subject to restriction that total price is within budget
    std::vector<Order<Offer>> choose_goods(
        double budget,
        const std::vector<std::shared_ptr<const Offer>>& offers
    );
    virtual std::vector<Order<Offer>> choose_goods(
        double budget,
        const std::vector<std::shared_ptr<const Offer>>& offers,
        // analogy to simulated annealing, more heat means more iterations for optimization
        int heat,
        // whether to shuffle offers before going over them, default is true
        bool shuffle
    );

    virtual void consume_goods() override;

    virtual void search_for_jobs() override;

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
    // analogous to choose_goods, but for jobs
    // in default implementation does not take utility of labor into account
    // i.e. only tries to maximize wages
    virtual std::vector<Order<JobOffer>> choose_jobs(
        double laborBudget,
        const std::vector<std::shared_ptr<const JobOffer>>& jobOffers
    );
    virtual std::vector<Order<JobOffer>> choose_jobs(
        double laborBudget,
        const std::vector<std::shared_ptr<const JobOffer>>& jobOffers,
        int heat,
        bool shuffle
    );
};

#endif
