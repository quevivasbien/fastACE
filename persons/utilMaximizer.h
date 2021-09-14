#ifndef UTILMAXIMIZER_H
#define UTILMAXIMIZER_H

#include "../economy.h"
#include "../functions/vectorInFloatOut.h"

class UtilMaximizer : public Person {
public:
    UtilMaximizer(Economy* economy);
    UtilMaximizer(Economy* economy, std::vector<GoodStock> inventory, double money, VectorInFloatOut utilityFunction);
    VectorInFloatOut utilityFunction;
    double u(const std::vector<double>& quantities) {
        return utilityFunction.f(quantities, numGoods);
    }
private:
    unsigned int numGoods;
};

#endif
