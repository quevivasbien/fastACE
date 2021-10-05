#include <vector>
#include <memory>
#include "base.h"

std::shared_ptr<Firm> Firm::create(Economy* economy, std::shared_ptr<Agent> owner) {
    std::shared_ptr<Firm> firm = std::shared_ptr<Firm>(new Firm(economy, owner));
    economy->add_firm(firm);
    return firm;
}

std::shared_ptr<Firm> Firm::create(
    Economy* economy,
    std::vector<std::shared_ptr<Agent>> owners,
    std::vector<double> inventory,
    double money
) {
    std::shared_ptr<Firm> firm = std::shared_ptr<Firm>(new Firm(economy, owners, inventory, money));
    economy->add_firm(firm);
    return firm;
}

Firm::Firm(Economy* economy, std::shared_ptr<Agent> owner)
    : Agent(economy) {
        owners.push_back(owner);
    }

Firm::Firm(Economy* economy, std::vector<std::shared_ptr<Agent>> owners, Eigen::ArrayXd inventory, double money)
    : Agent(economy, inventory, money), owners(owners) {}


bool Firm::time_step() {
    if (!Agent::time_step()) {
        return false;
    }
    else {
        search_for_laborers();
        produce();
        flush_myJobOffers();
        return true;
    }
}

void Firm::post_jobOffer(std::shared_ptr<JobOffer> jobOffer) {
    assert(jobOffer->get_offerer() == shared_from_this());
    economy->add_jobOffer(jobOffer);
    myJobOffers.push_back(jobOffer);
}

void Firm::check_my_jobOffers() {
    for (auto jobOffer : myJobOffers) {
        if (jobOffer->is_available()) {
            review_jobOffer_responses(jobOffer);
        }
    }
}

void Firm::flush_myJobOffers() {
    // figure out which myJobOffers are no longer available
    std::vector<unsigned int> idxs;
    for (unsigned int i = 0; i < myJobOffers.size(); i++) {
        if (!myJobOffers[i]->is_available()) {
            idxs.push_back(i);
        }
    }
    // remove those myJobOffers
    for (auto i : idxs) {
        myJobOffers[i] = myJobOffers.back();
        myJobOffers.pop_back();
    }
}
