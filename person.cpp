#include <vector>
#include "economy.h"

Person::Person(Economy* economy_) : Agent(economy_) {}

Person::Person(
    Economy* economy_, std::vector<GoodStock> inventory_, double money_
) : Agent(economy_, inventory_, money_) {}


bool Person::acceptJob(unsigned int i) {
    JobOffer jobOffer = economy->laborMarket[i];
    // check if the offer is still available, worker still has time for it, and employer can pay the wage
    if (!(jobOffer.claimed && (labor + jobOffer.labor > 1))) {
        jobOffer.claimed = true;
        if (jobOffer.offerer->getMoney() >= jobOffer.wage) {
            Job newJob {
                this,
                labor
            };
            jobOffer.offerer->addJob(newJob);
            // update worker's money and available labor
            money += jobOffer.wage;
            labor += jobOffer.labor;
            return true;
        }
    }
    return false;
}

void Person::searchForJob() {
    // as a toy example, just take available jobs until they can't fit
    unsigned int i = 0;
    while ((labor < 1) && (i < economy->laborMarket.size())) {
        acceptJob(i);
        i++;
    }
}

void Person::consumeGoods() {
    // as a toy example, just consume everything in inventory
    inventory.clear();
}
