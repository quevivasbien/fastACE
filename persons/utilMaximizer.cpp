#include "../economy.h"
#include "../functions/vectorInFloatOut.h"
#include "utilMaximizer.h"

const double defaultCDtfp = 1.0;
const std::vector<double> defaultCDParams = {0.5, 0.5};

UtilMaximizer::UtilMaximizer(
    Economy* economy
) : Person(economy), utilityFunction(CobbDouglas(defaultCDtfp, defaultCDParams)) {}

UtilMaximizer::UtilMaximizer(
    Economy* economy, std::vector<GoodStock> inventory, double money, VectorInFloatOut utilityFunction
) : Person(economy, inventory, money), utilityFunction(utilityFunction) {}

double UtilMaximizer::u(const std::vector<double>& quantities) {
    return utilityFunction.f(quantities, numGoods);
}
