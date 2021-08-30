#include "agent.h"

/*
class Agent {
public:
    void buyGoods();
    void consumeGoods();
    void offerLabor();
private:
    Economy* economy;  // the consumer needs to be able to look at the other members of the economy
    std::vector<GoodStock> inventory;
    double money;
}; /**/

Agent::Agent(Economy* economy_) : economy(economy_), money(0.0) {}

Agent::Agent(
    Economy* economy_, std::vector<GoodStock> inventory_, double money_
) : economy(economy_), inventory(inventory_), money(money_) {}


Agent::buyGoods() {}

Agent::consumeGoods() {}

Agent::offerLabor() {}
