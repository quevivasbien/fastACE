#include "utilMaxer.h"

const double defaultCDtfp = 1.0;
const Eigen::Array<double, 2, 1> defaultCDParams = {0.5, 0.5};

std::shared_ptr<UtilMaxer> UtilMaxer::create(Economy* economy) {
    auto utilMaxer = std::shared_ptr<UtilMaxer>(new UtilMaxer(economy));
    economy->add_person(utilMaxer);
    return utilMaxer;
}

std::shared_ptr<UtilMaxer> UtilMaxer::create(
    Economy* economy,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToScalar> utilFunc
) {
    auto utilMaxer = std::shared_ptr<UtilMaxer>(new UtilMaxer(
        economy, inventory, money, utilFunc
    ));
    economy->add_person(utilMaxer);
    return utilMaxer;
}

UtilMaxer::UtilMaxer(
    Economy* economy
) : Person(economy), utilFunc(std::make_shared<CobbDouglas>(defaultCDtfp, defaultCDParams)) {}

UtilMaxer::UtilMaxer(
    Economy* economy,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToScalar> utilFunc
) : Person(economy, inventory, money), utilFunc(utilFunc) {}

double UtilMaxer::u(const Eigen::ArrayXd& quantities) {
    return utilFunc->f(quantities);
}
