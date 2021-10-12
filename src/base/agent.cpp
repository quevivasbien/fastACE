#include "base.h"


Agent::Agent(Economy* economy) : economy(economy), money(0), time(economy->get_time()) {
    inventory = Eigen::ArrayXd::Zero(economy->get_numGoods());
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
        check_my_offers();
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

bool check_inventory_delta(
    const Eigen::ArrayXd& inventoryLeft,
    const Eigen::ArrayXd& delta
) {
    bool ok = true;
    for (unsigned int i = 0; i < inventoryLeft.size(); i++) {
        if (delta(i) > inventoryLeft(i)) {
            ok = false;
        }
    }
    return ok;
}

void update_offer_amount_left(
    Eigen::ArrayXd& inventoryLeft,
    const Eigen::ArrayXd& offerQuants,
    unsigned int& offerAmtLeft  // will be updated in place
) {
    // delta is the amount of goods needed to fulfill this offer
    Eigen::ArrayXd delta = offerQuants * offerAmtLeft;
    unsigned int reduction = 0;
    bool ok = false;
    while (!ok) {
        ok = check_inventory_delta(inventoryLeft, delta);
        if (!ok) {
            delta -= offerQuants;
            offerAmtLeft--;
        }
    }
    inventoryLeft -= delta;
}

void Agent::check_my_offers() {
    // default implementation just checks whether this agent can actually still fulfill all posted offers
    // inventoryLeft keeps track of how much of each good would be left after filling offers
    Eigen::ArrayXd inventoryLeft = inventory;
    for (auto offer : myOffers) {
        // changes inventoryLeft and offer->amountLeft in place
        update_offer_amount_left(
            inventoryLeft, offer->quantities, offer->amountLeft
        );
    }
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
            myCopy->amountLeft = 0;
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
    offer->amountLeft--;
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
