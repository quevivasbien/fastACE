#ifndef BASE_H
#define BASE_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <assert.h>

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


class Response {
public:
    Response(
        std::shared_ptr<BaseOffer> offer,
        std::shared_ptr<Agent> responder,
        unsigned int time
    );

    virtual ~Response() {}

    const std::shared_ptr<BaseOffer> get_offer() const;
    const std::shared_ptr<Agent> get_responder() const;
    unsigned int get_time() const;
    virtual bool is_available() const;

private:
    std::shared_ptr<BaseOffer> offer;
    std::shared_ptr<Agent> responder;
    unsigned int time;
};


class BaseOffer : public std::enable_shared_from_this<BaseOffer> {
    // Base class from which Offer and JobOffer inherit
    // TODO: figure out how to keep agents from messing with other people's stuff
public:
    BaseOffer(
        std::shared_ptr<Agent> offerer,
        unsigned int amount_available
    );

    virtual ~BaseOffer() {}

    unsigned int amount_left;
    const std::shared_ptr<Agent> get_offerer() const;
    const std::vector<std::shared_ptr<Response>>& get_responses() const;
    std::shared_ptr<Response> add_response(std::shared_ptr<Agent> responder);
    // unavailable offers will be swept up by the parent economy
    virtual bool is_available();
    unsigned int get_time_created() const;
protected:
    unsigned int time_created;
    // the agent who posted the offer
    std::shared_ptr<Agent> offerer;
    // the agents who have responded to the offer, indicating they want it
    std::vector<std::shared_ptr<Response>> responses;
};


class Offer : public BaseOffer {
public:
    Offer(
        std::shared_ptr<Agent> offerer,
        unsigned int amount_available,
        std::vector<unsigned int> good_ids,
        std::vector<double> quantities,
        double price
    );
    const std::vector<double>& get_quantities() const;
    const std::vector<unsigned int>& get_good_ids() const;
    virtual double get_price() const;
protected:
    std::vector<unsigned int> good_ids;
    std::vector<double> quantities;
    double price;
};


class JobOffer : public BaseOffer {
public:
    JobOffer(
        std::shared_ptr<Firm> offerer,
        unsigned int amount_available,
        double labor,
        double wage
    );
    double get_labor();
    virtual double get_wage() const;
protected:
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
    const std::vector<std::shared_ptr<Offer>>& get_market() const;
    const std::vector<std::shared_ptr<JobOffer>>& get_laborMarket() const;

    void add_offer(std::shared_ptr<Offer> offer);
    void add_jobOffer(std::shared_ptr<JobOffer> jobOffer);

protected:
    std::vector<std::shared_ptr<Person>> persons;
    std::vector<std::shared_ptr<Firm>> firms;
    // the names of goods for sale
    /// normally these goods will be referred to by their indices in the goods list
    std::vector<std::string> goods;
    unsigned int numGoods;  // equal to goods.size()
    std::vector<std::shared_ptr<Offer>> market;
    std::vector<std::shared_ptr<JobOffer>> laborMarket;
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
    // called via accept_offer_response, by the responder, finalizes a transaction if possible
    // won't do anything if the responder doesn't have the offer response in myResponses
    bool finalize_offer(std::shared_ptr<Response> response);

protected:
    Agent(Economy* economy);
    Agent(Economy* economy, std::vector<double> inventory, double money);

    Economy* economy;  // the economy this Agent is a part of
    std::vector<double> inventory;
    // the offers this agent has listed on the market
    std::vector<std::shared_ptr<Offer>> myOffers;
    // the responses this agent has made to other agents' offers
    std::vector<std::shared_ptr<Response>> myResponses;
    double money;
    unsigned int time;

    void add_to_inventory(unsigned int good_id, double quantity);
    void add_money(double amount);
    // loops through offers on market and decides whether to respond to each of them
    virtual void buy_goods();
    // lists offers for goods and checks responses to myOffers
    virtual void sell_goods();
    // looks at an an offer on the market and decides whether the agent wants it
    // (calls respond_to_offer(offer) if the agent wants it)
    virtual void look_at_offer(std::shared_ptr<Offer> offer) {};  // currently does nothing
    virtual void respond_to_offer(std::shared_ptr<Offer> offer);
    // decide whether to post new offers,
    // if yes, should call post_offer (possibly more than once)
    virtual void post_new_offers() {};  // currently does nothing
    // add offer to economy->market
    void post_offer(std::shared_ptr<Offer> offer);
    // Checks responses to all offers currently in myOffers
    void check_my_offers();
    // Looks at current responses to an offer from myOffers and decides whether to accept or reject
    virtual void review_offer_responses(std::shared_ptr<Offer> offer) {};  // currently does nothing
    // called by the offerer, finalizes a transaction if possible
    bool accept_offer_response(std::shared_ptr<Response> response);
    // creates a new firm with this agent as the first owner
    virtual void create_firm();
    // clear unavailable offers or responses
    void flush_myOffers();
    void flush_myResponses();
};


class Person : public Agent {
    // Persons are Agents which can also consume their goods and offer labor to Firms
public:
    template <typename T, typename ... Args>
	friend std::shared_ptr<T> create(Args&& ... args);

    std::shared_ptr<Person> get_shared_person();

    bool time_step() override;

protected:
    Person(Economy* economy);
    Person(Economy* economy, std::vector<double> inventory, double money);

    // the amount of labor this agent is currently using
    // typically cannot exceed 1.0
    double labor = 0.0;
    // the responses this person has made to firms' job offers
    std::vector<std::shared_ptr<Response>> myJobResponses;

    virtual void search_for_job();
    virtual void consume_goods() {};  // currently does nothing
    // looks at a job offer and decides whether to respond (apply)
    virtual void look_at_jobOffer(std::shared_ptr<JobOffer> jobOffer) {};  // currently does nothing
    virtual void respond_to_jobOffer(std::shared_ptr<JobOffer> jobOffer);
    // clear unavailable job responses
    void flush_myJobResponses();

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

    std::shared_ptr<Firm> get_shared_firm();

    bool time_step() override;

protected:
    Firm(Economy* economy, std::shared_ptr<Agent> owner);
    Firm(Economy* economy, std::vector<std::shared_ptr<Agent>> owners, std::vector<double> inventory, double money);

    std::vector<std::shared_ptr<Agent>> owners;
    double money;
    // the job offers this firm has listed on the job market
    std::vector<std::shared_ptr<JobOffer>> myJobOffers;
    void check_my_jobOffers();
    virtual bool review_jobOffer_responses(std::shared_ptr<JobOffer> jobOffer) {return false;};  // currently does nothing
    void post_jobOffer(std::shared_ptr<JobOffer> jobOffer);
    // clear unavailable job offers
    void flush_myJobOffers();
};


#endif
