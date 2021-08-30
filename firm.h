#ifndef FIRM_H
#define FIRM_H

#include <vector>

#include "economy.h"


class Firm {
public:
    void buyFactors();
    void hireLaborers();  // handled differently than other factors
    void produce();
    void payDividends();
private:
    Economy* economy;
    std::vector<Agent*> owners;
    std::vector<GoodStock> inventory;  // includes both factors and products
    double money;
};

#endif
