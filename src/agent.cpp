#include <string>
#include <vector>
#include "economy.h"


Agent::Agent(Economy* economy) : economy(economy), money(0) {}

Agent::Agent(
    Economy* economy, std::vector<GoodStock> inventory, double money
) : economy(economy), inventory(inventory), money(money) {}


void Agent::add_to_inventory(std::string good, double quantity) {
    bool dont_have = true;
    for (unsigned int i = 0; i < inventory.size(); i++) {
        if (inventory[i].good == good) {
            inventory[i].quantity += quantity;
            dont_have = false;
            break;
        }
    }
    if (dont_have) {
        GoodStock newGoodStock {
            good,
            quantity
        };
        inventory.push_back(newGoodStock);
    }
}

bool Agent::remove_from_inventory(std::string good, double quantity) {
    // removes quantity of good from inventory, if that quantity is available
    for (unsigned int i = 0; i < inventory.size(); i++) {
        if (inventory[i].good == good) {
            double difference = inventory[i].quantity - quantity;
            if (difference < 0) {
                return false;  // unsuccessful if difference < 0
            }
            else {
                inventory[i].quantity = difference;
                return true;
            }
        }
    }
    return false;
}

float Agent::get_money() {
    return money;
}

void Agent::add_money(float amount) {
    money += amount;
}

void Agent::flush_inventory() {
    // deletes all empty goods from inventory
    unsigned int i = 0;
    while (i < inventory.size()) {
        if (inventory[i].quantity == 0) {
            inventory.erase(inventory.begin() + i);
        }
        else {
            i++;
        }
    }
}

bool Agent::accept_offer(unsigned int i) {
    Offer offer = economy->market[i];
    // check if the offer is still available and is affordable before claiming
    if (!(offer.claimed || (money < offer.price))) {
        offer.claimed = true;
        // take the good if the seller actually has it
        bool transactionSuccess = offer.offerer->remove_from_inventory(offer.good, offer.quantity);
        if (transactionSuccess) {
            // add to buyer inventory
            add_to_inventory(offer.good, offer.quantity);
            // exchange money
            money -= offer.price;
            offer.offerer->money += offer.price;
            return true;  // successful
        }
    }
    return false;  // unsuccessful
}

bool Agent::create_offer(std::string good, double quantity, double price) {
    // check that the agent actually has the needed good and quantity before adding to market
    bool hasGood = false;
    for (unsigned int i = 0; i < inventory.size(); i++) {
        if (inventory[i].good == good) {
            if (inventory[i].quantity >= quantity) {
                hasGood = true;
            }
            break;
        }
    }
    if (hasGood) {
        Offer newOffer {
            this,
            good,
            quantity,
            price
        };
        economy->market.push_back(newOffer);
        return true;  // successful
    }
    else {
        return false;  // unsuccessful
    }
}

void Agent::buy_goods() {
    // as a toy example, just go through the available goods and buy until out of money
    for (unsigned int i = 0; i < economy->market.size(); i++) {
        if (money >= 0) {
            accept_offer(i);
        }
        else {
            break;
        }
    }
}

void Agent::sell_goods() {
    // as a toy example, just offer all goods in inventory for sale for price 1
    for (unsigned int i = 0; i < inventory.size(); i++) {
        create_offer(inventory[i].good, inventory[i].quantity, 1.0);
    }
}
