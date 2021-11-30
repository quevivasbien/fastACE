#include "base.h"


BaseOffer::BaseOffer(
    std::weak_ptr<Agent> offerer,
    unsigned int amount_available
) : offerer(offerer), amountLeft(amount_available) {}

bool BaseOffer::is_available() const {
    return (amountLeft > 0);
}


Offer::Offer(
    std::weak_ptr<Agent> offerer,
    unsigned int amount_available,
    Eigen::ArrayXd quantities,
    double price
) : BaseOffer(offerer, amount_available), quantities(quantities), price(price) {}


JobOffer::JobOffer(
    std::weak_ptr<Firm> offerer,
    unsigned int amount_available,
    double labor,
    double wage
) : BaseOffer(offerer, amount_available), labor(labor), wage(wage) {}
