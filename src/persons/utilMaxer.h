#ifndef UTILMAXIMIZER_H
#define UTILMAXIMIZER_H

#include <memory>
#include <vector>
#include <string>
#include "base.h"
#include "vecToScalar.h"
#include "solve.h"


class UtilMaxer : public Person {
public:
    UtilMaxer(Economy* economy);
    UtilMaxer(Economy* economy, std::vector<double> inventory, double money, std::shared_ptr<VecToScalar> utilFunc);
    double u(const Vec& quantities);  // alias for utilFunc.f
private:
    std::shared_ptr<VecToScalar> utilFunc;
};

#endif
