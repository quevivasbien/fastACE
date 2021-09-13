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
    std::vector<Offer> market;
    std::vector<JobOffer> laborMarket;

protected:
    void flushMarket();  // clear claimed offers
    void flushLaborMarket();  // clear claimed job offers

};


class Agent {
    // Agents are the most basic object in the economy.
    // They can buy and sell goods, keep inventories, and hold money.
public:
    Agent(Economy* economy_);
    Agent(Economy* economy_, std::vector<GoodStock> inventory_, double money_);
    virtual void buyGoods();
    virtual void sellGoods();
    void addToInventory(std::string good, double quantity);
    bool removeFromInventory(std::string good, double quantity);
    float getMoney();
    void addMoney(float amount);

protected:
    Economy* economy;  // the economy this Agent is a part of
    std::vector<GoodStock> inventory;
    double money;
    void flushInventory();  // remove any goods with zero quantity from the inventory
    bool createOffer(std::string good, double quantity, double price);
    bool acceptOffer(unsigned int i);  // accept the offer from economy->market with index i
};


class Person : public Agent {
    // Persons are Agents which can also consume their goods and offer labor to Firms
public:
    Person(Economy* economy_);
    Person(Economy* economy_, std::vector<GoodStock> inventory_, double money_);
    virtual void consumeGoods();
    virtual void searchForJob();
    bool acceptJob(unsigned int i);

protected:
    double labor = 0;

};


class Firm : public Agent {
    // Firms can hire laborers (Persons), produce new goods, and pay dividends on profits
    // Firms are owned by other Agents (other firms or persons)
public:
    Firm(Economy* economy_, std::shared_ptr<Agent> owner);
    Firm(Economy* economy_, std::vector<std::shared_ptr<Agent>> owners_, std::vector<GoodStock> inventory_, double money_);
    virtual void hireLaborers();
    virtual void produce();
    virtual void payDividends();
    void addJob(Job job);

protected:
    std::vector<std::shared_ptr<Agent>> owners;
    std::vector<Job> jobs;
    double money;
};


#endif
