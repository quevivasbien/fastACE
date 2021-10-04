#include "base.h"

std::shared_ptr<Agent> Agent::create(Economy* economy) {
    return std::shared_ptr<Agent>(new Agent(economy));
}

std::shared_ptr<Agent> Agent::create(
    Economy* economy, std::vector<double> inventory, double money
) {
    return std::shared_ptr<Agent>(new Agent(economy, inventory, money));
}

Agent::Agent(Economy* economy) : economy(economy), money(0), time(economy->get_time()) {
    inventory = std::vector<double>(economy->get_numGoods());
}

Agent::Agent(
    Economy* economy, std::vector<double> inventory, double money
) : economy(economy), inventory(inventory), money(money), time(economy->get_time()) {
    assert(inventory.size() == economy->get_numGoods());
}


bool Agent::time_step() {
    if (time != economy->get_time()) {
        time++;
        buy_goods();
        sell_goods();
        flush_myOffers();
        flush_myResponses();
        return true;  // completed successfully
    }
    else {
        return false;  // did not step
    }
}

unsigned int Agent::get_time() const { return time; };
Economy* Agent::get_economy() const { return economy; }
double Agent::get_money() const { return money; }


void Agent::buy_goods() {
    const std::vector<std::shared_ptr<Offer>> market = economy->get_market();
    for (auto offer : market) {
        if (offer->is_available()) {
            look_at_offer(offer);
        }
    }
}

void Agent::sell_goods() {
    post_new_offers();
    check_my_offers();
}


void Agent::add_to_inventory(unsigned int good_id, double quantity) {
    inventory[good_id] += quantity;
}

void Agent::add_money(double amount) {
    money += amount;
}

void Agent::post_offer(std::shared_ptr<Offer> offer) {
    // check that the offerer is the person listing it
    assert(offer->get_offerer() == shared_from_this());
    economy->add_offer(offer);
    myOffers.push_back(offer);
}

void Agent::respond_to_offer(std::shared_ptr<Offer> offer) {
    std::shared_ptr<Response> response = offer->add_response(shared_from_this());
    myResponses.push_back(response);
}

void Agent::check_my_offers() {
    for (auto offer : myOffers) {
        if (offer->is_available()) {
            review_offer_responses(offer);
        }
    }
}

bool Agent::accept_offer_response(std::shared_ptr<Response> response) {
    std::shared_ptr<Offer> offer = std::static_pointer_cast<Offer>(response->get_offer());
    // first check that offer is owned by this agent
    assert(offer->get_offerer() == shared_from_this());
    // make sure it's still available and response is at least 1 period old before proceeding
    if (!(offer->is_available() && response->get_time() > offer->get_time_created())) {
        return false;
    }
    // make sure this agent actually has enough goods
    const std::vector<double> quantities = offer->get_quantities();
    for (auto i : offer->get_good_ids()) {
        if (inventory[i] < quantities[i]) {
            // mark for removal and return false
            offer->amount_left = 0;
            return false;
        }
    }
    // send to responder for finalization
    if (response->get_responder()->finalize_offer(response)) {
        // complete transaction
        money += offer->get_price();
        for (auto i : offer->get_good_ids()) {
            inventory[i] -= offer->get_quantities()[i];
        }
        // change listing to -= 1 amount available
        offer->amount_left -= 1;
        // signal transaction successful
        return true;
    }
    else {
        return false;
    }
}

bool Agent::finalize_offer(std::shared_ptr<Response> response) {
    // check that the agent actually responded to this offer
    bool responded = false;
    for (auto myResponse : myResponses) {
        if (myResponse == response) {
            responded = true;
            break;
        }
    }
    if (!responded) {
        return false;
    }
    std::shared_ptr<Offer> offer = std::static_pointer_cast<Offer>(response->get_offer());
    // check that the agent has enough money
    if (money >= offer->get_price()) {
        // complete transaction
        money -= offer->get_price();
        std::vector<double> quantities = offer->get_quantities();
        for (auto i : offer->get_good_ids()) {
            inventory[i] += offer->get_quantities()[i];
        }
        // signal transaction successful
        return true;
    }
    else {
        return false;
    }
}

void Agent::create_firm() {
    economy->add_firm(shared_from_this());
}

void Agent::flush_myOffers() {
    // figure out which myOffers are no longer available
    std::vector<unsigned int> idxs;
    for (unsigned int i = 0; i < myOffers.size(); i++) {
        if (!myOffers[i]->is_available()) {
            idxs.push_back(i);
        }
    }
    // remove those myOffers
    for (auto i : idxs) {
        myOffers[i] = myOffers.back();
        myOffers.pop_back();
    }
}

void Agent::flush_myResponses() {
    // figure out which myResponses are no longer available
    std::vector<unsigned int> idxs;
    for (unsigned int i = 0; i < myResponses.size(); i++) {
        if (!myResponses[i]->is_available()) {
            idxs.push_back(i);
        }
    }
    // remove those myResponses
    for (auto i : idxs) {
        myResponses[i] = myResponses.back();
        myResponses.pop_back();
    }
}
