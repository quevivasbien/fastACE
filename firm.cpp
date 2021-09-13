#include <memory>
#include "economy.h"

Firm::Firm(Economy* economy_, std::shared_ptr<Agent> owner)
    : Agent(economy_), owners(std::vector<std::shared_ptr<Agent>> {owner}) {}

Firm::Firm(Economy* economy_, std::vector<std::shared_ptr<Agent>> owners_, std::vector<GoodStock> inventory_, double money_)
    : Agent(economy_, inventory_, money_), owners(owners_) {}


void Firm::hireLaborers() {
    // as a toy example, create a single job listing for 1 unit of labor, with wage being whatever money the firm has
    JobOffer newJobOffer {
        this,
        1,
        money
    };
    economy->laborMarket.push_back(newJobOffer);
}

void Firm::produce() {
    // we can define a production function here
    // as a toy example, say that the firm can produce one "defaultGood" with every unit of labor
    double totalLabor = 0;
    for (unsigned int i = 0; i < jobs.size(); i++) {
        totalLabor += jobs[i].labor;
    }
    addToInventory("defaultGood", totalLabor);
}

void Firm::payDividends() {
    // as a toy example, evenly divide money between all owners
    double moneyPerOwner = money / owners.size();
    for (unsigned int i = 0; i < owners.size(); i++) {
        owners[i]->addMoney(moneyPerOwner);
    }
    money = 0;
}

void Firm::addJob(Job job) {
    jobs.push_back(job);
}
