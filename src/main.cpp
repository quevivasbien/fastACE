#include <memory>
#include "neuralScenarios.h"

using namespace neural;

int main() {

    auto scenario = neural::SimpleScenario();
    auto economy = scenario.setup();

    for (int t = 0; t < 2; t++) {
        economy->time_step();
        scenario.trainer->train_on_episode();
    }

    // auto offerEncoder = std::make_shared<OfferEncoder>(
    //     DEFAULT_STACK_SIZE,
    //     20,
    //     DEFAULT_HIDDEN_SIZE,
    //     DEFAULT_ENCODING_SIZE
    // );

    // auto purchaseNet = std::make_shared<PurchaseNet>(
    //     offerEncoder,
    //     2,
    //     3,
    //     DEFAULT_HIDDEN_SIZE
    // );

    // auto purchaseNet2 = std::make_shared<PurchaseNet>(
    //     offerEncoder,
    //     2,
    //     3,
    //     DEFAULT_HIDDEN_SIZE
    // );

    // auto offerFeatures = torch::rand({DEFAULT_STACK_SIZE, 20});
    // auto utilParams = torch::rand(2);
    // auto money = torch::rand(1);
    // auto labor = torch::rand(1);
    // auto inventory = torch::rand(3);

    // auto encodedOffers = offerEncoder->forward(offerFeatures);
    // auto out = purchaseNet->forward(encodedOffers, utilParams, money, labor, inventory);

    // auto out2 = purchaseNet2->forward(encodedOffers, utilParams, money, labor, inventory);

    // std::cout << out << std::endl;
    // std::cout << out2 << std::endl;

    // auto loss = out.pow(2).mean();
    // loss.backward({}, true);

    // // for (auto p : purchaseNet->parameters()) {
    // //     std::cout << p.grad() << std::endl;
    // // }

    // auto loss2 = out2.pow(2).mean();
    // loss2.backward();

    return 0;
}
