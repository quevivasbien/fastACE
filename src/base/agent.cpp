#include "base.h"


Agent::Agent(Economy* economy) : economy(economy), money(0), time(economy->get_time()) {
    inventory = Eigen::ArrayXd(economy->get_numGoods());
}

Agent::Agent(
    Economy* economy, Eigen::ArrayXd inventory, double money
) : economy(economy), inventory(inventory), money(money), time(economy->get_time()) {
    assert(inventory.size() == economy->get_numGoods());
}


bool Agent::time_step() {
    if (time != economy->get_time()) {
        time++;
        labor = 0.0;
        buy_goods();
        sell_goods();
        flush_myOffers();
        return true;  // completed successfully
    }
    else {
        return false;  // did not step
    }
}

unsigned int Agent::get_time() const { return time; };
Economy* Agent::get_economy() const { return economy; }
double Agent::get_money() const { return money; }
const Eigen::ArrayXd& Agent::get_inventory() const { return inventory; }


void Agent::buy_goods() {
    const std::vector<std::shared_ptr<const Offer>> market = economy->get_market();
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
    assert(offer->offerer == shared_from_this());
    economy->add_offer(offer);
    myOffers.push_back(offer);
}

bool Agent::respond_to_offer(std::shared_ptr<const Offer> offer) {
    // check that the agent actually has enough money, then send to offerer
    if (money >= offer->price) {
        bool accepted = offer->offerer->review_offer_response(shared_from_this(), offer);
        if (accepted) {
            // complete the transaction on this end
            money -= offer->price;
            for (auto i : offer->good_ids) {
                inventory(i) += offer->quantities(i);
            }
            // transaction successful
            return true;
        }
    }
    // unsuccessful
    return false;
}

bool Agent::review_offer_response(std::shared_ptr<Agent> responder, std::shared_ptr<const Offer> offer) {
    // check that the offer is in myOffers
    std::shared_ptr<Offer> myCopy;
    for (auto myOffer : myOffers) {
        if (myOffer == offer) {
            myCopy = myOffer;
            break;
        }
    }
    if (myCopy == nullptr) {
        return false;
    }
    // need to use myCopy from here since it's not const
    // make sure this agent can actually afford the transaction
    for (auto i : myCopy->good_ids) {
        if (inventory(i) < myCopy->quantities(i)) {
            // mark for removal and return false
            myCopy->amount_left = 0;
            return false;
        }
    }
    // all good, let's go!
    accept_offer_response(myCopy);
    return true;
}

void Agent::accept_offer_response(std::shared_ptr<Offer> offer) {
    money += offer->price;
    for (auto i : offer->good_ids) {
        inventory(i) -= offer->quantities(i);
    }
    // change listing to -= 1 amount available
    offer->amount_left--;
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


void Agent::print_summary() {
    std::cout << "----------" << std::endl
        << "Memory ID: " << this << std::endl
        << "----------" << std::endl;
    std::cout << "Economy: " << economy << std::endl;
    std::cout << "Time: " << time << std::endl << std::endl;
    std::cout << "Inventory:" << std::endl;
    for (unsigned int i = 0; i < economy->get_numGoods(); i++) {
        const std::string* good_name = economy->get_name_for_good_id(i);
        std::cout << *good_name << ": " << inventory(i) << std::endl;
    }
    std::cout << std::endl << "Money: " << money << std::endl << std::endl;
}
