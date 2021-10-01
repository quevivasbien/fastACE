#include <vector>
#include <memory>
#include "base.h"

Firm::Firm(Economy* economy, std::shared_ptr<Agent> owner)
    : Agent(economy) {
        owners.push_back(owner);
        economy->add_firm(get_shared_firm());
    }

Firm::Firm(Economy* economy, std::vector<std::shared_ptr<Agent>> owners, std::vector<double> inventory, double money)
    : Agent(economy, inventory, money), owners(owners) {
        economy->add_firm(get_shared_firm());
    }


std::shared_ptr<Firm> Firm::get_shared_firm() {
    return std::static_pointer_cast<Firm>(shared_from_this());
}


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
    flush_offers<JobOffer>(myJobOffers);
}
