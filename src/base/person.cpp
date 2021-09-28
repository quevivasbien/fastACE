#include <vector>
#include "economy.h"

Person::Person(Economy* economy) : Agent(economy) {}

Person::Person(
    Economy* economy, std::vector<GoodStock> inventory, double money
) : Agent(economy, inventory, money) {}


bool Person::accept_job(unsigned int i) {
    JobOffer jobOffer = economy->laborMarket[i];
    // check if the offer is still available, worker still has time for it, and employer can pay the wage
    if (!(jobOffer.claimed && (labor + jobOffer.labor > 1))) {
        jobOffer.claimed = true;
        if (jobOffer.offerer->get_money() >= jobOffer.wage) {
            Job newJob {
                this,
                labor
            };
            jobOffer.offerer->add_job(newJob);
            // update worker's money and available labor
            money += jobOffer.wage;
            labor += jobOffer.labor;
            return true;
        }
    }
    return false;
}

void Person::search_for_job() {
    // as a toy example, just take available jobs until they can't fit
    unsigned int i = 0;
    while ((labor < 1) && (i < economy->laborMarket.size())) {
        accept_job(i);
        i++;
    }
}

void Person::consume_goods() {
    // as a toy example, just consume everything in inventory
    inventory.clear();
}
