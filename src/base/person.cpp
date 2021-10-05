#include <vector>
#include "base.h"
#include <iostream>

Person::Person(Economy* economy) : Agent(economy) {}

Person::Person(
    Economy* economy, Eigen::ArrayXd inventory, double money
) : Agent(economy, inventory, money) {}


bool Person::time_step() {
    if (!Agent::time_step()) {
        return false;
    }
    else {
        search_for_job();
        consume_goods();
        return true;
    }
}

void Person::search_for_job() {
    for (auto jobOffer : economy->get_laborMarket()) {
        look_at_jobOffer(jobOffer);
    }
}

bool Person::respond_to_jobOffer(std::shared_ptr<const JobOffer> jobOffer) {
    // check that the person actually has enough labor remaining, then send to offerer
    if (labor + jobOffer->labor <= 1) {
        bool accepted = std::static_pointer_cast<Firm>(jobOffer->offerer)->review_jobOffer_response(
            std::static_pointer_cast<Person>(shared_from_this()), jobOffer
        );
        if (accepted) {
            labor += jobOffer->labor;
            money += jobOffer->wage;
            return true;
        }
    }
    return false;
}
