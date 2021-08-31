#ifndef FIRM_H
#define FIRM_H

#include "agent.h"


class Firm : public Agent {
    // Firms can hire laborers (Persons), produce new goods, and pay dividends on profits
    // Firms are owned by other Agents (other firms or persons)
public:
    void hireLaborers();
    void produce();
    void payDividends();

protected:
    std::vector<Agent*> owners;
    std::vector<Job> jobs;
    double money;
};

#endif
