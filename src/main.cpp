#include "base.h"
#include "utilMaxer.h"
#include "profitMaxer.h"

const int numPeople = 20;

int main() {
    std::vector<std::string> goods = {"bread", "capital"};
    Economy economy(goods);

    Eigen::ArrayXd initWealths = 10 * (1.0 + Eigen::ArrayXd::Random(numPeople));
    Eigen::ArrayXd alphas = 0.5 * (1.0 + Eigen::ArrayXd::Random(numPeople));

    std::vector<std::shared_ptr<Agent>> people;
    for (unsigned int i = 0; i < numPeople; i++) {
        Eigen::Array2d elasticities(alphas[i], 0.0);
        auto person = UtilMaxer::init(
            &economy,
            Eigen::ArrayXd::Zero(2),
            initWealths(i),
            std::make_shared<CobbDouglas>(1.0, elasticities),
            std::make_shared<BasicPersonDecisionMaker>()
        );
        people.push_back(person);
    }

    Eigen::Array3d productivities(0.5, 0.0, 0.9);
    Eigen::Array2d firmInventory(3.0, 3.0);
    std::vector<std::shared_ptr<VecToVec>> innerFunctions = {
        std::make_shared<VToVFromVToS>(
            std::make_shared<Linear>(productivities), 2, 0
        ),
        std::make_shared<VToVFromVToS>(
            std::make_shared<Linear>(productivities), 2, 1
        )
    };
    auto prodFunc = std::make_shared<SumOfVecToVec>(innerFunctions);
    auto firmDecisionMaker = std::make_shared<BasicFirmDecisionMaker>();

    auto firm = ProfitMaxer::init(&economy, people, firmInventory, 100.0, prodFunc, firmDecisionMaker);

    for (unsigned int t = 1; t <= 10000; t++) {
        economy.print_summary();
        // for (auto person : people) {
        //     person->print_summary();
        // }
        firm->print_summary();
        economy.time_step();
    }

    return 0;
}
