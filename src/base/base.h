#ifndef BASE_H
#define BASE_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <assert.h>
#include <Eigen/Dense>

class BaseOffer;
class Agent;
class Person;
class Firm;


// a helper function template for instantiating Agent objects
// should be included as a friend function in any class that inherits from Agent
template <typename T, typename ... Args>
std::shared_ptr<T> create(Args&& ... args) {
	std::shared_ptr<T> agent = std::shared_ptr<T>(new T(std::forward<Args>(args) ...));
    agent->economy->add_agent(agent);
    return agent;
}


struct BaseOffer {
    // Base struct from which Offer and JobOffer inherit
    BaseOffer(
        std::shared_ptr<Agent> offerer,
        unsigned int amount_available
    );
    virtual ~BaseOffer() {}

    unsigned int amountLeft;
    // the agent who posted the offer
    std::shared_ptr<Agent> offerer;

    // unavailable offers will be swept up by the parent economy
    // in most cases just returns whether amountLeft > 0
    virtual bool is_available() const;
};


struct Offer : BaseOffer {
public:
    Offer(
        std::shared_ptr<Agent> offerer,
        unsigned int amount_available,
        std::vector<unsigned int> good_ids,
        Eigen::ArrayXd quantities,
        double price
    );

    std::vector<unsigned int> good_ids;
    Eigen::ArrayXd quantities;
    double price;
};


struct JobOffer : BaseOffer {
public:
    JobOffer(
        std::shared_ptr<Firm> offerer,
        unsigned int amount_available,
        double labor,
        double wage
    );

    double labor;
    double wage;
};

// TODO: consider implementing contracts for goods and especially labor
// esp if search costs are implemented


class Economy {
    // the Economy manages all the agents and holds the markets for goods and labor
public:
    Economy(std::vector<std::string> goods);

    virtual ~Economy() {}

    virtual bool time_step();
    unsigned int get_time() const { return time; };

    virtual std::shared_ptr<Person> add_person();
    virtual void add_agent(std::shared_ptr<Person> person);
    virtual std::shared_ptr<Firm> add_firm(std::shared_ptr<Agent> firstOwner);
    virtual void add_agent(std::shared_ptr<Firm> firm);

    const std::string* get_name_for_good_id(unsigned int id) const;

    const std::vector<std::shared_ptr<Person>>& get_persons() const;
    const std::vector<std::shared_ptr<Firm>>& get_firms() const;
    const std::vector<std::string>& get_goods() const;
    unsigned int get_numGoods() const;
    const std::vector<std::shared_ptr<const Offer>>& get_market() const;
    const std::vector<std::shared_ptr<const JobOffer>>& get_laborMarket() const;

    void add_offer(std::shared_ptr<const Offer> offer);
    void add_jobOffer(std::shared_ptr<const JobOffer> jobOffer);

protected:
    std::vector<std::shared_ptr<Person>> persons;
    std::vector<std::shared_ptr<Firm>> firms;
    // the names of goods for sale
    /// normally these goods will be referred to by their indices in the goods list
    std::vector<std::string> goods;
    unsigned int numGoods;  // equal to goods.size()
    std::vector<std::shared_ptr<const Offer>> market;
    std::vector<std::shared_ptr<const JobOffer>> laborMarket;
    // variable to keep track of time and control when economy can make a time_step()
    unsigned int time = 0;

    void flush_market();  // clear claimed offers
    void flush_labor_market();  // clear claimed job offers

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
    virtual bool review_offer_response(std::shared_ptr<Agent> responder, std::shared_ptr<const Offer> offer);

    // print a summary of this agent's current status
    virtual void print_summary();

protected:
    Agent(Economy* economy);
    Agent(Economy* economy, Eigen::ArrayXd inventory, double money);

    Economy* economy;  // the economy this Agent is a part of
    Eigen::ArrayXd inventory;
    // the offers this agent has listed on the market
    std::vector<std::shared_ptr<Offer>> myOffers;
    double money;
    unsigned int time;


    // the amount of labor this agent is currently using
    // for agents, cannot exceed 1.0
    // for firms, it's the amount of labor hired from Persons
    // typically reset to zero at the beginning of every period
    double labor = 0.0;

    void add_to_inventory(unsigned int good_id, double quantity);
    void add_money(double amount);
    // loops through offers on market and decides whether to respond to each of them
    virtual void buy_goods() {}  // by default does nothing
    // lists offers for goods and checks current offers to decide whether to keep them on the market
    virtual void sell_goods() {} // by default does nothing
    // looks at an an offer on the market and decides whether the agent wants it
    // (calls respond_to_offer(offer) if the agent wants it)
    virtual bool respond_to_offer(std::shared_ptr<const Offer> offer);
    // add offer to economy->market
    void post_offer(std::shared_ptr<Offer> offer);
    // Checks current offers to decide whether to keep them on the market
    virtual void check_my_offers();
    // called by the offerer during review_offer_response, finalizes a transaction
    void accept_offer_response(std::shared_ptr<Offer> offer);
    // creates a new firm with this agent as the first owner
    virtual void create_firm();
    // clear unavailable offers or responses
    void flush_myOffers();
};


class Person : public Agent {
    // Persons are Agents which can also consume their goods and offer labor to Firms
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> create(Args&& ... args);

    bool time_step() override;

protected:
    Person(Economy* economy);
    Person(Economy* economy, Eigen::ArrayXd inventory, double money);

    virtual void search_for_job();
    virtual void consume_goods() {};  // currently does nothing
    // looks at a job offer and decides whether to respond (apply)
    virtual void look_at_jobOffer(std::shared_ptr<const JobOffer> jobOffer) {};  // currently does nothing
    virtual bool respond_to_jobOffer(std::shared_ptr<const JobOffer> jobOffer);

};


class Firm : public Agent {
    // Firms can hire laborers (Persons), produce new goods, and pay dividends on profits
    // Firms are owned by other Agents (other firms or persons)
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> create(Args&& ... args);

    virtual void search_for_laborers() {};  // currently does nothing
    virtual void produce() {};  // currently does nothing
    virtual void pay_dividends() {};  // currently does nothing

    // analagous to Agent::review_offer_response
    virtual bool review_jobOffer_response(
        std::shared_ptr<Person> responder,
        std::shared_ptr<const JobOffer> jobOffer
    );

    bool time_step() override;

protected:
    Firm(Economy* economy, std::shared_ptr<Agent> owner);
    Firm(Economy* economy, std::vector<std::shared_ptr<Agent>> owners, Eigen::ArrayXd inventory, double money);

    std::vector<std::shared_ptr<Agent>> owners;
    // the job offers this firm has listed on the job market
    std::vector<std::shared_ptr<JobOffer>> myJobOffers;

    // analogous to Agent::check_my_offers
    void check_my_jobOffers() {}  // currently does nothing
    void accept_jobOffer_response(std::shared_ptr<JobOffer> jobOffer);
    void post_jobOffer(std::shared_ptr<JobOffer> jobOffer);
    // clear unavailable job offers
    void flush_myJobOffers();
};


#endif
