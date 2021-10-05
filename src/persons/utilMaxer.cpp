#include "utilMaxer.h"


UtilMaxer::UtilMaxer(
    Economy* economy
) : Person(economy), utilFunc(std::make_shared<CobbDouglas>()) {}

UtilMaxer::UtilMaxer(
    Economy* economy,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToScalar> utilFunc
) : Person(economy, inventory, money), utilFunc(utilFunc) {}

double UtilMaxer::u(const Eigen::ArrayXd& quantities) {
    return utilFunc->f(quantities);
}
