#ifndef AGENT_H
#define AGENT_H

#include <vector>
#include <string>
#include "economy.h"


class Agent {
    // Agents are the most basic object in the economy.
    // They can buy and sell goods, keep inventories, and hold money.
public:
    Agent(Economy* economy_);
    Agent(Economy* economy_, std::vector<GoodStock> inventory_, double money_);
    virtual void buyGoods();
    virtual void sellGoods();
    void addToInventory(std::string good, double quantity);
    bool removeFromInventory(std::string good, double quantity);

protected:
    Economy* economy;  // the economy this Agent is a part of
    std::vector<GoodStock> inventory;
    double money;
    void flushInventory();  // remove any goods with zero quantity from the inventory
    bool createOffer(std::string good, double quantity, double price);
    bool acceptOffer(unsigned int i);  // accept the offer from economy->market with index i
};

#endif
