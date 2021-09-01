#ifndef ECONOMY_H
#define ECONOMY_H

#include <vector>
#include <string>
#include <memory>


class Agent;
class Person;
class Firm;

struct GoodStock {
    std::string good;
    double quantity;
};

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
};

struct Job {
    Person* laborer;
    double labor;
};

// consider implementing contracts for goods and especially labor
// esp if search costs are implemented


class Economy {

public:
    std::vector<std::shared_ptr<Person>> persons;
    std::vector<std::shared_ptr<Firm>> firms;
    virtual void addPerson();
    virtual void addFirm(std::shared_ptr<Person> firstOwner);
    virtual void timeStep();

protected:
    std::vector<Offer> market;
    std::vector<JobOffer> laborMarket;
    void flushMarket();  // clear claimed offers
    void flushLaborMarket();  // clear claimed job offers

};

#endif
