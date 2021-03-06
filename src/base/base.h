#ifndef BASE_H
#define BASE_H

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <Eigen/Dense>
#include "util.h"


class Agent;
class Person;
class Firm;


class BaseOffer {
    // Base class from which Offer and JobOffer inherit
public:
    BaseOffer(
        std::weak_ptr<Agent> offerer,
        unsigned int amount_available
    );
    virtual ~BaseOffer() {}

    unsigned int amountLeft;
	unsigned int amountTaken = 0;
    // the agent who posted the offer
    std::weak_ptr<Agent> offerer;

    // unavailable offers will be swept up by the parent economy
    // in most cases just returns whether amountLeft > 0
    virtual bool is_available() const;
};


class Offer : public BaseOffer {
public:
    Offer(
        std::weak_ptr<Agent> offerer,
        unsigned int amount_available,
        Eigen::ArrayXd quantities,
        double price
    );

    Eigen::ArrayXd quantities;
    double price;
};


class JobOffer : public BaseOffer {
public:
    JobOffer(
        std::weak_ptr<Firm> offerer,
        unsigned int amount_available,
        double labor,
        double wage
    );

    double labor;
    double wage;
};


template <typename T>
struct Order {
    Order(
        std::weak_ptr<const T> offer,
        unsigned int amount
    ) : offer(offer), amount(amount) {}
    std::weak_ptr<const T> offer;
    unsigned int amount;
};

// TODO: consider implementing contracts for goods and especially labor
// esp if search costs are implemented


class Economy {
    // the Economy manages all the agents and holds the markets for goods and labor
public:
    Economy(std::vector<std::string> goods);

    virtual ~Economy() {}

    virtual bool time_step();
    unsigned int get_time() const;

    std::shared_ptr<Person> add_person();
    std::shared_ptr<Firm> add_firm();
    void add_agent(std::shared_ptr<Person> person);
    void add_agent(std::shared_ptr<Firm> firm);

    const std::string& get_name_for_good_id(unsigned int id) const;

    const std::vector<std::weak_ptr<Person>>& get_persons() const;
    const std::vector<std::weak_ptr<Firm>>& get_firms() const;
    const std::vector<std::string>& get_goods() const;
    unsigned int get_numGoods() const;
    const std::vector<std::weak_ptr<const Offer>>& get_market() const;
    const std::vector<std::weak_ptr<const JobOffer>>& get_jobMarket() const;
    std::default_random_engine get_rng() const;

    void add_offer(std::weak_ptr<const Offer> offer);
    void add_jobOffer(std::weak_ptr<const JobOffer> jobOffer);
    
    virtual std::string get_typename() const;
    virtual void print_summary() const;

protected:
    std::vector<std::shared_ptr<Person>> persons;
    std::vector<std::shared_ptr<Firm>> firms;
    // persons_weak and firms_weak are to make sharing agents lists easier
    // without sharing ownership
    std::vector<std::weak_ptr<Person>> persons_weak;
    std::vector<std::weak_ptr<Firm>> firms_weak;
    // the names of goods for sale
    /// normally these goods will be referred to by their indices in the goods list
    std::vector<std::string> goods;
    unsigned int numGoods;  // equal to goods.size()
    std::vector<std::weak_ptr<const Offer>> market;
    std::vector<std::weak_ptr<const JobOffer>> jobMarket;
    std::default_random_engine rng;
    // variable to keep track of time and control when economy can make a time_step()
    unsigned int time = 0;

    std::mutex mutex;
};


class Agent : public std::enable_shared_from_this<Agent> {
    // Agents are the most basic member of the economy.
    // They can buy and sell goods, keep inventories, and hold money.
public:
    // Note: Agents can create shared pointers to themselves, but
    // this means that you _must_ create an Agent as a shared pointer
    // you should use the `create` template function to instantiate any class that inherits from Agent
    virtual ~Agent() {}

    // make a time step. returns true if completed successfully, else false
    virtual bool time_step();

    unsigned int get_time() const;
    Economy* get_economy() const;
    double get_money() const;
    const Eigen::ArrayXd& get_inventory() const;

    // Looks at current response to an offer from myOffers and decides whether to accept or reject
    // won't do anything if the responder doesn't have the offer in myOffers
    // returns true if offer is accepted & successful
    // *should call accept_offer_response to determine if it is successful
    // default implementation accepts all valid responses
    virtual bool review_offer_response(std::shared_ptr<Agent> responder, std::weak_ptr<const Offer> offer_);

    virtual std::string get_typename() const;
    // print a summary of this agent's current status
    virtual void print_summary() const;

protected:
    Agent(Economy* economy);
    Agent(Economy* economy, Eigen::ArrayXd inventory, double money);

    Economy* economy;  // the economy this Agent is a part of
    Eigen::ArrayXd inventory;
    // the offers this agent has listed on the market
    std::vector<std::shared_ptr<Offer>> myOffers;
    double money;
    unsigned int time;

    std::mutex myMutex;


    void add_to_inventory(unsigned int good_id, double quantity);
    void add_money(double amount);
    // loops through offers on market and decides whether to respond to each of them
    virtual void buy_goods() {}  // by default does nothing
    // lists offers for goods
    virtual void sell_goods() {} // by default does nothing
    // indicates that this agent wants an offer
    virtual bool respond_to_offer(std::weak_ptr<const Offer> offer);
    // add offer to economy->market and myOffers
    void post_offer(std::shared_ptr<Offer> offer);
    // Checks current offers to decide whether to keep them on the market
    virtual void check_my_offers();
    // called by the offerer during review_offer_response, finalizes a transaction
    void accept_offer_response(std::shared_ptr<Offer> offer);
};


class Person : public Agent {
    // Persons are Agents which can also consume their goods and offer labor to Firms
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> util::create(Args&& ... args);

    template <typename ... Args>
    static std::shared_ptr<Person> init(Args&& ... args) {
        return util::create<Person>(std::forward<Args>(args) ...);
    }

	double get_laborSupplied() const;
    bool time_step() override;

    virtual std::string get_typename() const override;

protected:
    Person(Economy* economy);
    Person(Economy* economy, Eigen::ArrayXd inventory, double money);

	double laborSupplied = 0.0;

    virtual void search_for_jobs() {}  // currently does nothing
    virtual void consume_goods() {}  // currently does nothing
    virtual bool respond_to_jobOffer(std::weak_ptr<const JobOffer> jobOffer);

};


class Firm : public Agent {
    // Firms can hire laborers (Persons), produce new goods, and pay dividends on profits
    // Firms are owned by other Agents (other firms or persons)
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> util::create(Args&& ... args);

    template <typename ... Args>
    static std::shared_ptr<Firm> init(Args&& ... args) {
        return util::create<Firm>(std::forward<Args>(args) ...);
    }

    virtual void search_for_laborers() {}  // currently does nothing
    virtual void produce() {}  // currently does nothing
    virtual void pay_dividends() {}  // currently does nothing

    // analagous to Agent::review_offer_response
    virtual bool review_jobOffer_response(
        std::shared_ptr<Person> responder,
        std::weak_ptr<const JobOffer> jobOffer
    );

	double get_laborHired() const;
    bool time_step() override;

    virtual std::string get_typename() const override;

protected:
    Firm(Economy* economy);
    Firm(Economy* economy, std::vector<std::shared_ptr<Agent>> owners, Eigen::ArrayXd inventory, double money);

    std::vector<std::shared_ptr<Agent>> owners;
    // the job offers this firm has listed on the job market
    std::vector<std::shared_ptr<JobOffer>> myJobOffers;
	double laborHired = 0.0;

    // analogous to Agent::check_my_offers
    virtual void check_myJobOffers();
    void accept_jobOffer_response(std::shared_ptr<JobOffer> jobOffer);
    void post_jobOffer(std::shared_ptr<JobOffer> jobOffer);
};


#endif
