#include "firm.h"

Firm::Firm(Economy* economy_, Agent* owner) : economy(economy_), owners(std::vector<Agent*> {owner}), money(0) {}

Firm::Firm(Economy* economy_, std::vector<Agent*> owners_, std::vector<GoodStock> inventory_, double money_)
    : economy(economy_), owners(owners_), inventory(inventory_), money(money_) {}


void Firm::HireLaborers() {
    // as a toy example, create a single job listing for 1 unit of labor, with wage 1
    JobOffer newJobOffer {
        this,
        1,
        1
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
        owners[i]->money += moneyPerOwner;
    }
    money = 0;
}
