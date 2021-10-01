#include <vector>
#include "base.h"

Person::Person(Economy* economy) : Agent(economy) {
    economy->add_person(get_shared_person());
}

Person::Person(
    Economy* economy, std::vector<double> inventory, double money
) : Agent(economy, inventory, money) {
    economy->add_person(get_shared_person());
}


std::shared_ptr<Person> Person::get_shared_person() {
    return std::static_pointer_cast<Person>(shared_from_this());
}


bool Person::time_step() {
    if (!Agent::time_step()) {
        return false;
    }
    else {
        search_for_job();
        consume_goods();
        flush_myJobResponses();
        return true;
    }
}

void Person::search_for_job() {
    const std::vector<std::shared_ptr<JobOffer>> jobOffers = economy->get_laborMarket();
    for (auto jobOffer : jobOffers) {
        look_at_jobOffer(jobOffer);
    }
}

void Person::respond_to_jobOffer(std::shared_ptr<JobOffer> jobOffer) {
    jobOffer->add_response(Response(shared_from_this(), time));
    myJobResponses.push_back(jobOffer);
}

void Person::flush_myJobResponses() {
    flush_offers<JobOffer>(myJobResponses);
}
