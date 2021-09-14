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
    virtual void add_person();
    virtual void add_person(std::shared_ptr<Person> person);
    virtual void add_firm(std::shared_ptr<Person> firstOwner);
    virtual void add_firm(std::shared_ptr<Firm> firm);
    virtual void time_step();
    std::vector<Offer> market;
    std::vector<JobOffer> laborMarket;

protected:
    void flush_market();  // clear claimed offers
    void flush_labor_market();  // clear claimed job offers

};


class Agent {
    // Agents are the most basic object in the economy.
    // They can buy and sell goods, keep inventories, and hold money.
public:
    Agent(Economy* economy);
    Agent(Economy* economy, std::vector<GoodStock> inventory, double money);
    virtual void buy_goods();
    virtual void sell_goods();
    void add_to_inventory(std::string good, double quantity);
    bool remove_from_inventory(std::string good, double quantity);
    float get_money();
    void add_money(float amount);
    Economy* economy;  // the economy this Agent is a part of

protected:
    std::vector<GoodStock> inventory;
    double money;
    void flush_inventory();  // remove any goods with zero quantity from the inventory
    bool create_offer(std::string good, double quantity, double price);
    bool accept_offer(unsigned int i);  // accept the offer from economy->market with index i
};


class Person : public Agent {
    // Persons are Agents which can also consume their goods and offer labor to Firms
public:
    Person(Economy* economy);
    Person(Economy* economy, std::vector<GoodStock> inventory, double money);
    virtual void consume_goods();
    virtual void search_for_job();
    bool accept_job(unsigned int i);

protected:
    double labor = 0;

};


class Firm : public Agent {
    // Firms can hire laborers (Persons), produce new goods, and pay dividends on profits
    // Firms are owned by other Agents (other firms or persons)
public:
    Firm(Economy* economy, std::shared_ptr<Agent> owner);
    Firm(Economy* economy, std::vector<std::shared_ptr<Agent>> owners, std::vector<GoodStock> inventory, double money);
    virtual void hire_laborers();
    virtual void produce();
    virtual void pay_dividends();
    void add_job(Job job);

protected:
    std::vector<std::shared_ptr<Agent>> owners;
    std::vector<Job> jobs;
    double money;
};


#endif
