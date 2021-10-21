#include "base.h"
#include "utilMaxer.h"
#include "profitMaxer.h"


int main() {
    std::vector<std::string> goods = {"wheat", "milk"};
    Economy economy(goods);

    Eigen::Array2d personInventory(0.0, 0.0);
    auto utilFunc = std::make_shared<CobbDouglas>(2);
    auto personDecisionMaker = std::make_shared<BasicPersonDecisionMaker>();

    auto person = UtilMaxer::init(&economy, personInventory, 10.0, utilFunc, personDecisionMaker);

    std::vector<std::shared_ptr<Agent>> owners = {person};
    Eigen::Array2d firmInventory(3.0, 3.0);
    std::vector<std::shared_ptr<VecToVec>> innerFunctions = {
        std::make_shared<VToVFromVToS>(
            std::make_shared<Linear>(3), 2, 0
        ),
        std::make_shared<VToVFromVToS>(
            std::make_shared<Linear>(3), 2, 1
        )
    };
    auto prodFunc = std::make_shared<SumOfVecToVec>(innerFunctions);
    auto firmDecisionMaker = std::make_shared<BasicFirmDecisionMaker>();

    auto firm = ProfitMaxer::init(&economy, owners, firmInventory, 0.0, prodFunc, firmDecisionMaker);

    for (unsigned int t = 1; t <= 5; t++) {
        economy.print_summary();
        person->print_summary();
        firm->print_summary();
        economy.time_step();
    }

    return 0;
}
