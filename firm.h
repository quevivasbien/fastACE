#ifndef FIRM_H
#define FIRM_H

#include <vector>
#include <memory>
#include "agent.h"
#include "economy.h"


class Firm : public Agent {
    // Firms can hire laborers (Persons), produce new goods, and pay dividends on profits
    // Firms are owned by other Agents (other firms or persons)
public:
    Firm(Economy* economy_, std::shared_ptr<Agent> owner);
    Firm(Economy* economy_, std::vector<std::shared_ptr<Agent>> owners_, std::vector<GoodStock> inventory_, double money_);
    virtual void hireLaborers();
    virtual void produce();
    virtual void payDividends();

protected:
    std::vector<std::shared_ptr<Agent>> owners;
    std::vector<Job> jobs;
    double money;
};

#endif
