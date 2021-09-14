#include <iostream>
#include <memory>
#include "economy.h"

void Economy::add_person() {
    persons.push_back(std::make_shared<Person>(this));
}

void Economy::add_person(std::shared_ptr<Person> person) {
    // check that person is assigned to this economy
    if (person->economy == this) {
        persons.push_back(person);
    }
    else {
        std::cout << "Failed to add Person to Economy; You must assign Person to this Economy." << std::endl;
    }
}

void Economy::add_firm(std::shared_ptr<Person> firstOwner) {
    firms.push_back(std::make_shared<Firm>(this, firstOwner));
}

void Economy::add_firm(std::shared_ptr<Firm> firm) {
    // check that firm is assigned to this economy
    if (firm->economy == this) {
        firms.push_back(firm);
    }
    else {
        std::cout << "Failed to add Firm to Economy; you must assign Firm to this Economy." << std::endl;
    }
}

void Economy::flush_market() {
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

void Economy::flush_labor_market() {
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

void Economy::time_step() {
    for (unsigned int i = 0; i < firms.size(); i++) {
        firms[i]->hire_laborers();
    }
    for (unsigned int i = 0; i < persons.size(); i++) {
        persons[i]->search_for_job();
        persons[i]->buy_goods();
        persons[i]->consume_goods();
    }
    for (unsigned int i = 0; i < firms.size(); i++) {
        firms[i]->produce();
        firms[i]->sell_goods();
        firms[i]->pay_dividends();
    }
    flush_market();
    flush_labor_market();
}
