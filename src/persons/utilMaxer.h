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
    UtilMaxer(Economy* economy, Eigen::ArrayXd inventory, double money, std::shared_ptr<VecToScalar> utilFunc);

    std::shared_ptr<VecToScalar> utilFunc;
};

#endif
