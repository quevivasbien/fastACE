#ifndef ECONOMY_H
#define ECONOMY_H

#include <vector>
#include <string>


class Agent;
class Person;
class Firm;

struct GoodStock {
    std::string good;
    double quantity;
}

struct Offer {
    Agent* offerer;
    std::string good;
    double quantity;
    double price;
    // bool isPromise;  // if false, then offer disappears if claimed once
    bool claimed = false;
};

struct JobOffer {
    Firm* offerer;
    double labor;
    double wage;
    bool claimed = false;
}

struct Job {
    Person* laborer;
    double labor;
    double wage;
}

// consider implementing contracts for goods and especially labor
// esp if search costs are implemented


class Economy {
public:
    void timeStep();

protected:
    std::vector<Person> persons;
    std::vector<Firm> firms;
    std::vector<Offer> market;
    std::vector<JobOffer> laborMarket;
    void flushMarket();  // clear claimed offers
    void flushLaborMarket();  // clear claimed job offers

};

#endif
