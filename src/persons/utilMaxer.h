#ifndef UTILMAXIMIZER_H
#define UTILMAXIMIZER_H

#include <memory>
#include "economy.h"
#include "vecToScalar.h"

class UtilMaxer : public Person {
public:
    UtilMaxer(Economy* economy);
    UtilMaxer(Economy* economy, std::vector<GoodStock> inventory, double money, std::shared_ptr<VecToScalar> utilFunc);
    double u(const Vec& quantities);  // alias for utilFunc.f
private:
    std::shared_ptr<VecToScalar> utilFunc;
    unsigned int numGoods;
};

#endif
