#include <memory>
#include "decisionNetHandler.h"
#include "neuralPersonDecisionMaker.h"
#include "neuralFirmDecisionMaker.h"
#include "neuralEconomy.h"
#include "utilMaxer.h"
#include "profitMaxer.h"

const int stackSize = 2;
const int offerEncodingSize = 3;


int main() {

    neural::NeuralEconomy economy({"bread", "capital"}, 20);

    auto dnh = std::make_shared<neural::DecisionNetHandler>(&economy);

    auto person1 = UtilMaxer::init(
        &economy,
        Eigen::Array2d(10.0, 10.0),
        20.0,
        std::make_shared<CES>(1.0, Eigen::Array3d(0.5, 0.5, 0.5), 1.3),
        std::make_shared<neural::NeuralPersonDecisionMaker>(dnh)
    );
    auto person2 = UtilMaxer::init(
        &economy,
        Eigen::Array2d(10.0, 10.0),
        20.0,
        std::make_shared<CES>(1.0, Eigen::Array3d(0.2, 0.6, 0.4), 1.3),
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

    economy.time_step();
    economy.print_summary();

    auto offerNet = std::make_shared<neural::OfferNet>(20, 5, 5, 2, 20);

    auto offerEncodings = torch::rand({5, 20});
    auto utilParams = torch::rand({5});
    auto money = torch::rand({1});
    auto labor = torch::rand({1});
    auto inventory = torch::rand({2});

    return 0;
}
