#include "utilMaxer.h"

const double defaultCDtfp = 1.0;
const Eigen::Array<double, 2, 1> defaultCDParams = {0.5, 0.5};

UtilMaxer::UtilMaxer(
    Economy* economy
) : Person(economy), utilFunc(std::make_shared<CobbDouglas>(defaultCDtfp, defaultCDParams)) {}

UtilMaxer::UtilMaxer(
    Economy* economy,
    std::vector<double> inventory,
    double money,
    std::shared_ptr<VecToScalar> utilFunc
) : Person(economy, inventory, money), utilFunc(utilFunc) {}

double UtilMaxer::u(const Vec& quantities) {
    return utilFunc->f(quantities);
}
