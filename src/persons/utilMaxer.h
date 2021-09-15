#ifndef UTILMAXIMIZER_H
#define UTILMAXIMIZER_H

#include "../economy.h"
#include "../functions/vecToScalar.h"

class UtilMaxer : public Person {
public:
    UtilMaxer(Economy* economy);
    UtilMaxer(Economy* economy, std::vector<GoodStock> inventory, double money, VecToScalar utilFunc);
    VecToScalar utilFunc;
    double u(const std::vector<double>& quantities);  // alias for utilFunc.f
private:
    unsigned int numGoods;
};

#endif
