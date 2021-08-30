#ifndef AGENT_H
#define AGENT_H

#include <vector>

#include "economy.h"


class Agent {
public:
    void buyGoods();
    void consumeGoods();
    void offerLabor();
private:
    Economy* economy;  // the consumer needs to be able to look at the other members of the economy
    std::vector<GoodStock> inventory;
    double money;
};

#endif
