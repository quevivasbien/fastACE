#include "profitMaxer.h"

const double defaultCDtfp = 1.0;
const Eigen::Array<double, 2, 1> defaultCDParams = {0.5, 0.5};

std::shared_ptr<ProfitMaxer> ProfitMaxer::create(Economy* economy, std::shared_ptr<Agent> owner) {
    auto profitMaxer = std::shared_ptr<ProfitMaxer>(new ProfitMaxer(economy, owner));
    economy->add_firm(profitMaxer);
    return profitMaxer;
}

std::shared_ptr<ProfitMaxer> ProfitMaxer::create(
    Economy* economy,
    std::vector<std::shared_ptr<Agent>> owners,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToScalar> prodFunc
) {
    auto profitMaxer = std::shared_ptr<ProfitMaxer>(new ProfitMaxer(
        Economy* economy,
        std::vector<std::shared_ptr<Agent>> owners,
        Eigen::ArrayXd inventory,
        double money,
        std::shared_ptr<VecToScalar> prodFunc
    ));
    economy->add_firm(profitMaxer);
    return profitMaxer;
}

ProfitMaxer::ProfitMaxer(Economy* economy, std::shared_ptr<Agent> owner
) : Firm(economy, owner), prodFunc(std::make_shared<CobbDouglas>(defaultCDtfp, defaultCDParams)) {}

ProfitMaxer::ProfitMaxer(
    Economy* economy,
    std::vector<std::shared_ptr<Agent>> owners,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToScalar> prodFunc
) : Firm(economy, owners, inventory, money), prodFunc(prodFunc) {}

double ProfitMaxer::f(const Eigen::ArrayXd& quantities) {
    return prodFunc->f(quantities);
}
