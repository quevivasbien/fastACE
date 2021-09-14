#include <vector>
#include <memory>
#include "economy.h"

Firm::Firm(Economy* economy, std::shared_ptr<Agent> owner)
    : Agent(economy), owners(std::vector<std::shared_ptr<Agent>> {owner}) {}

Firm::Firm(Economy* economy, std::vector<std::shared_ptr<Agent>> owners, std::vector<GoodStock> inventory, double money)
    : Agent(economy, inventory, money), owners(owners) {}


void Firm::hire_laborers() {
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
    add_to_inventory("defaultGood", totalLabor);
}

void Firm::pay_dividends() {
    // as a toy example, evenly divide money between all owners
    double moneyPerOwner = money / owners.size();
    for (unsigned int i = 0; i < owners.size(); i++) {
        owners[i]->add_money(moneyPerOwner);
    }
    money = 0;
}

void Firm::add_job(Job job) {
    jobs.push_back(job);
}
