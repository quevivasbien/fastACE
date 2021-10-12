#include "profitMaxer.h"

ProfitMaxer::ProfitMaxer(Economy* economy, std::shared_ptr<Agent> owner, unsigned int outputIndex) :
    Firm(economy, owner),
    prodFunc(
        std::make_shared<VToVFromVToS>(
            std::make_shared<CobbDouglas>(economy->get_numGoods() + 1),
            economy->get_numGoods(),
            outputIndex
        )
    ) {}

ProfitMaxer::ProfitMaxer(
    Economy* economy,
    std::shared_ptr<Agent> owner,
    std::shared_ptr<VecToVec> prodFunc
) : Firm(economy, owner), prodFunc(prodFunc) {
    assert(prodFunc->get_numInputs() == economy->get_numGoods() + 1);
}

ProfitMaxer::ProfitMaxer(
    Economy* economy,
    std::vector<std::shared_ptr<Agent>> owners,
    Eigen::ArrayXd inventory,
    double money,
    std::shared_ptr<VecToVec> prodFunc
) : Firm(economy, owners, inventory, money), prodFunc(prodFunc) {
    assert(prodFunc->get_numInputs() == economy->get_numGoods() + 1);
}


double ProfitMaxer::f(const Eigen::ArrayXd& quantities) {
    return prodFunc->f(quantities);
}


void ProfitMaxer::produce() {
    Eigen::VectorXd inputs(prodFunc->get_numInputs());
    inputs << labor, inventory;
    inventory = f(inputs);
}
