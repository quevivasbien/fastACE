#include <memory>
#include "decisionNets.h"

int main() {

    Economy economy({"bread", "capital"});

    auto person1 = economy.add_person();
    auto person2 = economy.add_person();
    auto offer1 = std::make_shared<Offer>(person1, 2, Eigen::Array2d(1.0, 2.5), 1.0);
    auto offer2 = std::make_shared<Offer>(person2, 2, Eigen::Array2d(2.1, 1.0), 2.0);
    economy.add_offer(offer1);
    economy.add_offer(offer2);

    auto encoder = std::make_shared<OfferEncoder>(2, 5, 10, 3);

    auto dm = std::make_shared<NeuralDecisionMaker>(&economy, encoder);
    dm->update_encodedOffers();

    std::cout << dm->encodedOffers << std::endl;

    return 0;
}
