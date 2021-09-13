#include <iostream>
#include <memory>
#include "economy.h"

void Economy::addPerson() {
    persons.push_back(std::make_shared<Person>(this));
}

void Economy::addPerson(std::shared_ptr<Person> person) {
    // check that person is assigned to this economy
    if (person->economy == this) {
        persons.push_back(person);
    }
    else {
        std::cout << "Failed to add Person to Economy; You must assign Person to this Economy." << std::endl;
    }
}

void Economy::addFirm(std::shared_ptr<Person> firstOwner) {
    firms.push_back(std::make_shared<Firm>(this, firstOwner));
}

void Economy::addFirm(std::shared_ptr<Firm> firm) {
    // check that firm is assigned to this economy
    if (firm->economy == this) {
        firms.push_back(firm);
    }
    else {
        std::cout << "Failed to add Firm to Economy; you must assign Firm to this Economy." << std::endl;
    }
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
