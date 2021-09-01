#include "economy.h"

void Economy::addPerson() {
    persons.push_back(std::make_shared<Person>(this));
}

void Economy::addFirm(std::shared_ptr<Person> firstOwner) {
    firms.push_back(std::make_shared<Firm>(this, firstOwner));
}

void Economy::flushMarket() {
    unsigned int i = 0;
    while (i < market.size()) {
        if (market[i].claimed) {
            market.erase(market.begin() + i);
        }
        else {
            i++;
        }
    }
}

void Economy::flushLaborMarket() {
    unsigned int i = 0;
    while (i < laborMarket.size()) {
        if (laborMarket[i].claimed) {
            laborMarket.erase(laborMarket.begin() + i);
        }
        else {
            i++;
        }
    }
}

void Economy::timeStep() {
    for (unsigned int i = 0; i < firms.size(); i++) {
        firms[i]->hireLaborers();
    }
    for (unsigned int i = 0; i < persons.size(); i++) {
        persons[i]->searchForJob();
        persons[i]->buyGoods();
        persons[i]->consumeGoods();
    }
    for (unsigned int i = 0; i < firms.size(); i++) {
        firms[i]->produce();
        firms[i]->sellGoods();
        firms[i]->payDividends();
    }
    flushMarket();
    flushLaborMarket();
}
