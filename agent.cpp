#include "agent.h"


Agent::Agent(Economy* economy_) : economy(economy_), money(0) {}

Agent::Agent(
    Economy* economy_, std::vector<GoodStock> inventory_, double money_
) : economy(economy_), inventory(inventory_), money(money_) {}


void Agent::addToInventory(std::string good, double quantity) {
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
        }
        inventory.push_back(newGoodStock);
    }
}

bool Agent::removeFromInventory(std::string good, double quantity) {
    // removes quantity of good from inventory, if that quantity is available
    for (unsigned int i = 0; i < inventory.size(); i++) {
        if (inventory[i].good == good) {
            double difference = inventory[i].quantity - quantity;
            if (difference < 0) {
                return false;  // unsuccessful if difference < 0
            }
            else {
                inventory[i].good.quantity = difference;
                return true;
            }
        }
    }
}

void Agent::flushInventory() {
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

bool Agent::acceptOffer(unsigned int i) {
    Offer offer = economy->market[i];
    // check if the offer is still available and is affordable before claiming
    if !(offer.claimed || (money < offer.price)) {
        offer.claimed = true;
        // take the good if the seller actually has it
        bool transactionSuccess = offer.offerer->removeFromInventory(offer.good, offer.quantity);
        if (transactionSuccess) {
            // add to buyer inventory
            addToInventory(offer.good, offer.quantity);
            // exchange money
            money -= offer.price;
            offer.offerer->money += offer.price;
            return true;  // successful
        }
    }
    return false;  // unsuccessful
}

bool Agent::createOffer(std::string good, double quantity, double price) {
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

void Agent::buyGoods() {
    // as a toy example, just go through the available goods and buy until out of money
    for (unsigned int i = 0; i < economy->market.size(); i++) {
        if (money >= 0) {
            acceptOffer(i);
        }
        else {
            break;
        }
    }
}

void Agent::sellGoods() {
    // as a toy example, just offer all goods in inventory for sale
    for (unsigned int i = 0; i < inventory.size(); i++) {
        createOffer(inventory[i].good, inventory[i].quantity);
    }
}
