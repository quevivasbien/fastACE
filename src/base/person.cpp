#include "base.h"

Person::Person(std::shared_ptr<Economy> economy) : Agent(economy) {}

Person::Person(
    std::shared_ptr<Economy> economy, Eigen::ArrayXd inventory, double money
) : Agent(economy, inventory, money) {}

std::string Person::get_typename() const {
    return "Person";
}


double Person::get_laborSupplied() const {
    return laborSupplied;
}


bool Person::time_step() {
    if (!Agent::time_step()) {
        return false;
    }
    else {
        laborSupplied = 0.0;
        print_status(this, "Searching for job...");
        search_for_jobs();
        print_status(this, "Buying goods...");
        buy_goods();
        print_status(this, "Consuming goods...");
        consume_goods();
        return true;
    }
}


bool Person::respond_to_jobOffer(std::weak_ptr<const JobOffer> jobOffer_) {
    // check that the person actually has enough labor remaining, then send to offerer
    auto jobOffer = jobOffer_.lock();
    if (laborSupplied + jobOffer->labor <= 1) {
        print_status(this, "Asking for jobOffer acceptance...");
        bool accepted = std::static_pointer_cast<Firm>(
            jobOffer->offerer.lock()
        )->review_jobOffer_response(
            std::static_pointer_cast<Person>(shared_from_this()), jobOffer
        );
        if (accepted) {
            std::lock_guard<std::mutex> lock(myMutex);
            laborSupplied += jobOffer->labor;
            money += jobOffer->wage;
            return true;
        }
    }
    return false;
}
