#include <vector>
#include "base.h"
#include <iostream>

std::shared_ptr<Person> Person::create(Economy* economy) {
    std::shared_ptr<Person> person = std::shared_ptr<Person>(new Person(economy));
    economy->add_person(person);
    return person;
}

std::shared_ptr<Person> Person::create(Economy* economy, std::vector<double> inventory, double money) {
    std::shared_ptr<Person> person = std::shared_ptr<Person>(new Person(economy, inventory, money));
    economy->add_person(person);
    return person;
}

Person::Person(Economy* economy) : Agent(economy) {}

Person::Person(
    Economy* economy, std::vector<double> inventory, double money
) : Agent(economy, inventory, money) {}


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
    std::shared_ptr<Response> response = jobOffer->add_response(shared_from_this());
    myJobResponses.push_back(response);
}

void Person::flush_myJobResponses() {
    // figure out which myJobResponses are no longer available
    std::vector<unsigned int> idxs;
    for (unsigned int i = 0; i < myJobResponses.size(); i++) {
        if (!myJobResponses[i]->is_available()) {
            idxs.push_back(i);
        }
    }
    // remove those myJobResponses
    for (auto i : idxs) {
        myJobResponses[i] = myJobResponses.back();
        myJobResponses.pop_back();
    }
}
