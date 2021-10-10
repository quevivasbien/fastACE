#ifndef PROFITMAXER_H
#define PROFITMAXER_H

#include "base.h"
#include "vecToVec.h"
#include "solve.h"


class ProfitMaxer : public Firm {
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> create(Args&& ... args);

    double f(const Eigen::ArrayXd& quantities);
protected:
    ProfitMaxer(Economy* economy, std::shared_ptr<Agent> owner, unsigned int outputIndex);
    ProfitMaxer(
        Economy* economy,
        std::shared_ptr<Agent> owner,
        std::shared_ptr<VecToVec> prodFunc
    );
    ProfitMaxer(
        Economy* economy,
        std::vector<std::shared_ptr<Agent>> owners,
        Eigen::ArrayXd inventory,
        double money,
        std::shared_ptr<VecToVec> prodFunc
    );

    std::shared_ptr<VecToVec> prodFunc;
};

#endif
