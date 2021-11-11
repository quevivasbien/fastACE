#include <memory>
#include "decisionNetHandler.h"
#include "neuralPersonDecisionMaker.h"
#include "utilMaxer.h"

const int stackSize = 2;
const int offerEncodingSize = 3;


int main() {

    Economy economy({"bread", "capital"});

    auto dnh = std::make_shared<neural::DecisionNetHandler>(&economy);

    auto person1 = UtilMaxer::init(
        &economy,
        std::make_shared<CES>(1.0, Eigen::Array3d(0.5, 0.5, 0.5), 1.3),
        std::make_shared<neural::NeuralPersonDecisionMaker>(dnh)
    );
    auto person2 = UtilMaxer::init(
        &economy,
        std::make_shared<CES>(1.0, Eigen::Array3d(0.2, 0.6, 0.4), 1.3),
        std::make_shared<neural::NeuralPersonDecisionMaker>(dnh)
    );

    auto offer1 = std::make_shared<Offer>(person1, 2, Eigen::Array2d(1.0, 2.5), 1.0);
    auto offer2 = std::make_shared<Offer>(person2, 2, Eigen::Array2d(2.1, 1.0), 2.0);
    economy.add_offer(offer1);
    economy.add_offer(offer2);

    economy.time_step();

    return 0;
}
