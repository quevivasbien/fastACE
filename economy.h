#ifndef ECONOMY_H
#define ECONOMY_H

#include <vector>
#include <string>


class Agent;
class Firm;

struct Good {
    std::string name;
    // include other characteristics of the good here
};

struct GoodStock {
    Good* good;
    double quantity;
}

struct Offer {
    Good* good;
    double quantity;
    double price;
    bool isPromise;  // if false, then offer disappears if claimed once
};

struct LaborOffer {
    Agent* laborer;
    double wage;
}

// consider implementing contracts for goods and especially labor
// esp if search costs are implemented


class Economy {
private:
    std::vector<Agent> agents;
    std::vector<Firm> firms;
    std::vector<Offer> market;
    std::vector<LaborOffer> laborMarket;
};

#endif
