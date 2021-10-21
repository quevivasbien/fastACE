#include "base.h"
#include "utilMaxer.h"
#include "profitMaxer.h"


int main() {
    std::vector<std::string> goods = {"wheat", "milk"};
    Economy economy(goods);

    auto person = UtilMaxer::init(&economy);

    std::vector<std::shared_ptr<Agent>> owners = {person};
    Eigen::Array2d inventory(3.0, 3.0);
    std::vector<std::shared_ptr<VecToVec>> innerFunctions = {
        std::make_shared<VToVFromVToS>(
            std::make_shared<Linear>(3), 2, 0
        ),
        std::make_shared<VToVFromVToS>(
            std::make_shared<Linear>(3), 2, 1
        )
    };
    auto prodFunc = std::make_shared<SumOfVecToVec>(innerFunctions);
    auto decisionMaker = std::make_shared<BasicFirmDecisionMaker>();

    auto firm = ProfitMaxer::init(&economy, owners, inventory, 0.0, prodFunc, decisionMaker);

    person->print_summary();
    firm->print_summary();

    for (unsigned int t = 1; t <= 5; t++) {
        print(std::string("~~~ Time ="), std::to_string(t), std::string("~~~"));
        economy.time_step();
        person->print_summary();
        firm->print_summary();
    }

    return 0;
}
