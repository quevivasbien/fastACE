#include "person.h"

Person::Person(Economy* economy_) : economy(economy_), money(0) {}

Person::Person(
    Economy* economy_, std::vector<GoodStock> inventory_, double money_
) : economy(economy_), inventory(inventory_), money(money_) {}


bool Person::acceptJob(unsigned int i) {
    JobOffer jobOffer = economy->laborMarket[i];
    // check if the offer is still available and if worker still has time for it
    if !(jobOffer.claimed && (labor + jobOffer.labor > 1)) {
        jobOffer.claimed = true;
        Job newJob {
            this,
            labor,
            wage
        };
        jobOffer.offerer->jobs.push_back(newJob);
        return true;
    }
    else {
        return false;
    }
}

void Person::searchForJob() {
    // as a toy example, just take available jobs until they can't fit
    unsigned int i = 0;
    while ((labor < 1) && (i < laborMarket.size())) {
        acceptJob(i);
        i++;
    }
}

void Person::consumeGoods() {
    // as a toy example, just consume everything in inventory
    inventory.clear();
}
