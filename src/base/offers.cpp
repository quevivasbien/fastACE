#include "base.h"


Response::Response(
    std::shared_ptr<BaseOffer> offer,
    std::shared_ptr<Agent> responder,
    unsigned int time
) : offer(offer), responder(responder), time(time) {}

const std::shared_ptr<BaseOffer> Response::get_offer() const {return offer;}
const std::shared_ptr<Agent> Response::get_responder() const {return responder;}
unsigned int Response::get_time() const {return time;}

bool Response::is_available() const {
    return offer->is_available();
}


BaseOffer::BaseOffer(
    std::shared_ptr<Agent> offerer,
    unsigned int amount_available
) : offerer(offerer), amount_left(amount_available), time_created(offerer->get_time()) {}

const std::shared_ptr<Agent> BaseOffer::get_offerer() const { return offerer; }
const std::vector<std::shared_ptr<Response>>& BaseOffer::get_responses() const { return responses; }

std::shared_ptr<Response> BaseOffer::add_response(std::shared_ptr<Agent> responder) {
    auto response = std::make_shared<Response>(shared_from_this(), responder, responder->get_time());
    responses.push_back(response);
    return response;
}

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
