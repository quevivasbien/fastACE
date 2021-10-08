#ifndef UTILMAXER_H
#define UTILMAXER_H

#include "base.h"
#include "vecToScalar.h"
// #include "solve.h"


struct Order {
    Order(
        std::shared_ptr<const Offer> offer,
        unsigned int amount
    ) : offer(offer), amount(amount) {}
    std::shared_ptr<const Offer> offer,
    unsigned int amount
};


std::vector<std::shared_ptr<const Offer>> filterAvailable(
    const std::vector<std::shared_ptr<const Offer>>& offers,
    bool shuffle = true
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
    
    // helper functions for choose_goods:
    int find_best_offer(
        const std::vector<std::shared_ptr<const Offer>>& offers,
        unsigned int numOffers,
        double budgetLeft,
        std::vector<unsigned int> numTaken,
        const Eigen::ArrayXd& quantities
    );
    void fill_basket(
        const std::vector<std::shared_ptr<const Offer>>& availOffers,
        unsigned int numOffers,
        double& budgetLeft,
        std::vector<unsigned int>& numTaken,
        Eigen::ArrayXd& quantities
    );
    int find_worst_offer(
        const std::vector<std::shared_ptr<const Offer>>& offers,
        unsigned int numOffers,
        std::vector<unsigned int> numTaken,
        const Eigen::ArrayXd& quantities
    );
    void empty_basket(
        const std::vector<std::shared_ptr<const Offer>>& availOffers,
        unsigned int numOffers,
        double& budgetLeft,
        std::vector<unsigned int>& numTaken,
        Eigen::ArrayXd& quantities,
        int heat
    );
    
    // called by buy_goods()
    // looks at current goods on market and chooses bundle that maximizes utility
    // subject to restriction that total price is within budget
    virtual void choose_goods(
        double budget,
        const std::vector<std::shared_ptr<const Offer&> offers,
        int heat = 5,
        bool shuffle = true
    );
};

#endif
