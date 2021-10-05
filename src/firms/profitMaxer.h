#ifndef PROFITMAXER_H
#define PROFITMAXER_H

#include "base.h"
#include "vecToScalar.h"
#include "solve.h"


class ProfitMaxer : public Firm {
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> create(Args&& ... args);

    double f(const Eigen::ArrayXd& quantities);
protected:
    ProfitMaxer(Economy* economy, std::shared_ptr<Agent> owner);
    ProfitMaxer(Economy* economy, Eigen::ArrayXd inventory, double money, std::shared_ptr<VecToScalar> prodFunc);

    std::shared_ptr<VecToScalar> prodFunc;
};

#endif
