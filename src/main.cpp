#include <memory>
#include "neuralDecisionMaker.h"

const int stackSize = 2;
const int offerEncodingSize = 3;

int main() {

    // Economy economy({"bread", "capital"});
    //
    // auto person1 = economy.add_person();
    // auto person2 = economy.add_person();
    // auto offer1 = std::make_shared<Offer>(person1, 2, Eigen::Array2d(1.0, 2.5), 1.0);
    // auto offer2 = std::make_shared<Offer>(person2, 2, Eigen::Array2d(2.1, 1.0), 2.0);
    // economy.add_offer(offer1);
    // economy.add_offer(offer2);
    //
    //
    // int numGoods = economy.get_numGoods();
    // int numUtilParams = economy.get_numGoods();
    // int numAgents = economy.get_totalAgents();
    //
    // auto encoder = std::make_shared<OfferEncoder>(stackSize, numAgents + numGoods + 1, 10, offerEncodingSize);
    // auto purchaseNet = std::make_shared<PurchaseNet>(offerEncodingSize, stackSize, numUtilParams, numGoods, 10);
    //
    // auto dm = std::make_shared<NeuralDecisionMaker>(&economy, encoder, purchaseNet);
    //
    // std::vector<int> offerIndices = {0, 1};
    // Eigen::Array2d utilParams = {0.5, 0.5};
    // double budget = 1.0;
    // Eigen::Array2d inventory = person1->get_inventory();
    //
    // auto offers = dm->get_Offers_to_request(offerIndices, utilParams, budget, inventory);
    //
    // for (auto o : offers) {
    //     std::cout << o << std::endl;
    // }

    auto x = torch::randint(10, 20, torch::dtype(torch::kInt));
    std::cout << x << std::endl;

    return 0;
}
