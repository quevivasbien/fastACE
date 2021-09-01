#ifndef PERSON_H
#define PERSON_H

#include "agent.h"

class Person : public Agent {
    // Persons are Agents which can also consume their goods and offer labor to Firms
public:
    Person(Economy* economy_);
    Person(Economy* economy_, std::vector<GoodStock> inventory_, double money_);
    virtual void consumeGoods();
    virtual void searchForJob();

protected:
    double labor = 0;

};

#endif
