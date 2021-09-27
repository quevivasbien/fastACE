#include "economy.h"
#include "functions/vecToScalar.h"
#include "utilMaxer.h"

const double defaultCDtfp = 1.0;
const Eigen::Array<double, 2, 1> defaultCDParams = {0.5, 0.5};

UtilMaxer::UtilMaxer(
    Economy* economy
) : Person(economy), utilFunc(CobbDouglas(defaultCDtfp, defaultCDParams)) {}

UtilMaxer::UtilMaxer(
    Economy* economy, std::vector<GoodStock> inventory, double money, VecToScalar utilFunc
) : Person(economy, inventory, money), utilFunc(utilFunc) {}

double UtilMaxer::u(const Vec& quantities) {
    return utilFunc.f(quantities);
}
