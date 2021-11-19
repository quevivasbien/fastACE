#include <memory>
#include "decisionNetHandler.h"
#include "neuralPersonDecisionMaker.h"
#include "neuralFirmDecisionMaker.h"
#include "neuralEconomy.h"
#include "utilMaxer.h"
#include "profitMaxer.h"

int main() {

    neural::NeuralEconomy economy({"bread", "capital"}, 20);

    auto dnh = std::make_shared<neural::DecisionNetHandler>(&economy);

    auto person1 = UtilMaxer::init(
        &economy,
        Eigen::Array2d(10.0, 10.0),
        20.0,
        std::make_shared<CES>(1.0, Eigen::Array3d(0.5, 0.5, 0.5), 1.3),
        0.8,
        std::make_shared<neural::NeuralPersonDecisionMaker>(dnh)
    );
    auto person2 = UtilMaxer::init(
        &economy,
        Eigen::Array2d(10.0, 10.0),
        20.0,
        std::make_shared<CES>(1.0, Eigen::Array3d(0.2, 0.6, 0.4), 1.3),
        0.9,
        std::make_shared<neural::NeuralPersonDecisionMaker>(dnh)
    );

    auto firm = ProfitMaxer::init(
        &economy,
        std::vector<std::shared_ptr<Agent>>({person1}),
        Eigen::Array2d(10.0, 20.0),
        50.0,
        create_CES_VecToVec(
            std::vector<double>({0.5, 1.0}),
            std::vector<Eigen::ArrayXd>({
                Eigen::Array3d(1.0, 0.0, 1.0),
                Eigen::Array3d(1.0, 0.0, 1.0)
            }),
            std::vector<double>({3.0, 5.0})
        ),
        std::make_shared<neural::NeuralFirmDecisionMaker>(dnh)
    );

    for (int t = 0; t < 10; t++) {
        economy.time_step();
    }

    return 0;
}
