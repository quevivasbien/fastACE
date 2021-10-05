#include "base.h"


BaseOffer::BaseOffer(
    std::shared_ptr<Agent> offerer,
    unsigned int amount_available
) : offerer(offerer), amount_left(amount_available), time_created(offerer->get_time()) {}

bool BaseOffer::is_available() const {
    return (amount_left > 0);
}


Offer::Offer(
    std::shared_ptr<Agent> offerer,
    unsigned int amount_available,
    std::vector<unsigned int> good_ids,
    Eigen::ArrayXd quantities,
    double price
) : BaseOffer(offerer, amount_available), good_ids(good_ids), quantities(quantities), price(price) {}


JobOffer::JobOffer(
    std::shared_ptr<Firm> offerer,
    unsigned int amount_available,
    double labor,
    double wage
) : BaseOffer(offerer, amount_available), labor(labor), wage(wage) {}
