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
        check_my_offers();
        {
            std::lock_guard<std::mutex> lock(myMutex);
            flush<Offer>(myOffers);
        }
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
    std::lock_guard<std::mutex> lock(myMutex);
    inventory[good_id] += quantity;
}

void Agent::add_money(double amount) {
    std::lock_guard<std::mutex> lock(myMutex);
    money += amount;
}

void Agent::post_offer(std::shared_ptr<Offer> offer) {
    std::lock_guard<std::mutex> lock(myMutex);
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
    std::lock_guard<std::mutex> lock(myMutex);
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
        print_status(this, "Asking for offer acceptance...");
        bool accepted = offer->offerer->review_offer_response(shared_from_this(), offer);
        if (accepted) {
            std::lock_guard<std::mutex> lock(myMutex);
            // complete the transaction on this end
            money -= offer->price;
            inventory += offer->quantities;
            // transaction successful
            return true;
        }
    }
    // unsuccessful
    return false;
}

bool Agent::review_offer_response(std::shared_ptr<Agent> responder, std::shared_ptr<const Offer> offer) {
    std::shared_ptr<Offer> myCopy;
    {
        std::lock_guard<std::mutex> lock(myMutex);
        print_status(this, "Reviewing offer response...");
        if (!offer->is_available()) {
            print_status(this, "Requested offer is not available.");
            return false;
        }
        // check that the offer is in myOffers
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
        if ((inventory < myCopy->quantities).any()) {
            print_status(this, "I can't afford to fulfill this offer.");
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
    std::lock_guard<std::mutex> lock(myMutex);
    print_status(this, "Accepting offer response...");
    money += offer->price;
    inventory -= offer->quantities;
    // change listing to -= 1 amount available
    offer->amountLeft--;
    // mark that one of these has actually been sold
    offer->amountTaken++;
}


void Agent::create_firm() {
    economy->add_firm(shared_from_this());
}


std::string Agent::get_typename() const {
    return "Agent";
}


void Agent::print_summary() const {
    std::cout << "\n----------\n"
        << "Memory ID: " << this << " (" << get_typename() << ")\n"
        << "----------\n";
    std::cout << "Time: " << time << "\n\n";
    std::cout << "Inventory:\n";
    for (unsigned int i = 0; i < economy->get_numGoods(); i++) {
        const std::string* good_name = economy->get_name_for_good_id(i);
        std::cout << *good_name << ": " << inventory(i) << '\n';
    }
    std::cout << "\nMoney: " << money << "\n\n";
}
