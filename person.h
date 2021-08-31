#ifndef PERSON_H
#define PERSON_H

#include "agent.h"

class Person : public Agent {
    // Persons are Agents which can also consume their goods and offer labor to Firms
public:
    void consumeGoods();
    void searchForJob();

protected:
    bool searchForJob(unsigned int i);
    double labor = 0;
    
};

#endif
