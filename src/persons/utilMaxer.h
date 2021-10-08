#ifndef UTILMAXER_H
#define UTILMAXER_H

#include "base.h"
#include "vecToScalar.h"
#include "solve.h"


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

    // called by buy_goods()
    // looks at current goods on market and chooses bundle that maximizes utility
    // subject to restriction that total price is within budget
    virtual void choose_goods(double budget);
};

#endif
