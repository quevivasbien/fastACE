#include "base.h"



BaseOffer::BaseOffer(
    std::shared_ptr<Agent> offerer,
    unsigned int amount_available
) : offerer(offerer), amount_left(amount_available), time_created(offerer->get_time()) {}

const std::shared_ptr<Agent> BaseOffer::get_offerer() const { return offerer; }
const std::vector<Response>& BaseOffer::get_responses() const { return responses; }
void BaseOffer::add_response(Response response) { responses.push_back(response); }
bool BaseOffer::is_available() {
    // must have some amount remaining
    // & offers cannot receive responses until at least 1 time period after they're listed
    return (amount_left > 0) && (offerer->get_time() > time_created);
}
unsigned int BaseOffer::get_time_created() const { return time_created; }



Offer::Offer(
    std::shared_ptr<Agent> offerer,
    unsigned int amount_available,
    std::vector<unsigned int> good_ids,
    std::vector<double> quantities,
    double price
) : BaseOffer(offerer, amount_available), good_ids(good_ids), quantities(quantities), price(price) {}

const std::vector<double>& Offer::get_quantities() const { return quantities; }
const std::vector<unsigned int>& Offer::get_good_ids() const { return good_ids; }
double Offer::get_price() const { return price; }



JobOffer::JobOffer(
    std::shared_ptr<Firm> offerer,
    unsigned int amount_available,
    double labor,
    double wage
) : BaseOffer(offerer, amount_available), labor(labor), wage(wage) {}

double JobOffer::get_labor() { return labor; }
double JobOffer::get_wage() const { return wage; };



template<typename T>
void flush_offers(std::vector<std::shared_ptr<T>>& offers) {
    // figure out which offers are no longer available
    std::vector<unsigned int> idxs;
    for (unsigned int i = 0; i < offers.size(); i++) {
        if (!offers[i]->is_available()) {
            idxs.push_back(i);
        }
    }
    // remove those offers
    for (auto i : idxs) {
        offers[i] = offers.back();
        offers.pop_back();
    }
}
