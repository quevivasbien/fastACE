#ifndef AGENT_H
#define AGENT_H

#include "economy.h"


class Agent {
    // Agents are the most basic object in the economy.
    // They can buy and sell goods, keep inventories, and hold money.
public:
    void buyGoods();
    void sellGoods();
    void addToInventory(std::string good, double quantity);
    double removeFromInventory(std::string good, double quantity);

protected:
    Economy* economy;  // the economy this Agent is a part of
    std::vector<GoodStock> inventory;
    double money;
    void flushInventory();  // remove any goods with zero quantity from the inventory
    bool createOffer(std::string good, double quantity, price);
    bool acceptOffer(unsigned int i);  // accept the offer from economy->market with index i
};

#endif
