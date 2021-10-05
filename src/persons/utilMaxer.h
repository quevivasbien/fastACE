#ifndef UTILMAXER_H
#define UTILMAXER_H

#include "base.h"
#include "vecToScalar.h"
#include "solve.h"


class UtilMaxer : public Person {
public:
    static std::shared_ptr<UtilMaxer> create(Economy* economy);
    static std::shared_ptr<UtilMaxer> create(
        Economy* economy,
        Eigen::ArrayXd inventory,
        double money,
        std::shared_ptr<VecToScalar> utilFunc
    );
    double u(const Eigen::ArrayXd& quantities);  // alias for utilFunc.f
protected:
    UtilMaxer(Economy* economy);
    UtilMaxer(Economy* economy, Eigen::ArrayXd inventory, double money, std::shared_ptr<VecToScalar> utilFunc);

    std::shared_ptr<VecToScalar> utilFunc;
};

#endif
