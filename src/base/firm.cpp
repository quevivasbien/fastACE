#include <vector>
#include <memory>
#include <typeinfo>
#include "base.h"

Firm::Firm(Economy* economy)
    : Agent(economy) {}

Firm::Firm(Economy* economy, std::vector<std::shared_ptr<Agent>> owners, Eigen::ArrayXd inventory, double money)
    : Agent(economy, inventory, money), owners(owners) {}


std::string Firm::get_typename() const {
    return "Firm";
}


double Firm::get_laborHired() const {
    return laborHired;
}


bool Firm::time_step() {
    if (!Agent::time_step()) {
        return false;
    }
    else {
        util::print_status(this, "Checking my job offers...");
        check_myJobOffers();
        {
            std::lock_guard<std::mutex> lock(myMutex);
            util::flush(myJobOffers);
        }
        util::print_status(this, "Buying goods...");
        buy_goods();
        util::print_status(this, "Producing...");
        produce();
        util::print_status(this, "Selling goods...");
        sell_goods();
        pay_dividends();
        laborHired = 0.0;
        util::print_status(this, "Searching for laborers...");
        search_for_laborers();
        return true;
    }
}

void Firm::post_jobOffer(std::shared_ptr<JobOffer> jobOffer) {
    std::lock_guard<std::mutex> lock(myMutex);
    assert(jobOffer->offerer.lock() == shared_from_this());
    economy->add_jobOffer(std::weak_ptr<const JobOffer>(jobOffer));
    myJobOffers.push_back(jobOffer);
}


bool Firm::review_jobOffer_response(
    std::shared_ptr<Person> responder,
    std::weak_ptr<const JobOffer> jobOffer
) {
    std::shared_ptr<const JobOffer> jobOffer_ = jobOffer.lock();
    std::shared_ptr<JobOffer> myCopy = nullptr;
    {
        std::lock_guard<std::mutex> lock(myMutex);
        if (!jobOffer_->is_available()) {
            util::print_status(this, "Requested offer is not available.");
            return false;
        }
        // check that the offer is in myOffers
        for (auto myOffer : myJobOffers) {
            if (myOffer == jobOffer_) {
                myCopy = myOffer;
                break;
            }
        }
        if (myCopy == nullptr) {
            return false;
        }
        // need to use myCopy from here since it's not const
        // make sure this firm can actually afford to pay the wage
        if (money < myCopy->wage) {
            util::print_status(this, "I can't afford to fulfill this offer.");
            // mark for removal
            myCopy->amountLeft = 0;
            return false;
        }
    }
    // all good, let's go!
    accept_jobOffer_response(myCopy);
    return true;
}


void Firm::check_myJobOffers() {
    std::lock_guard<std::mutex> lock(myMutex);
    double moneyLeft = money;
    for (auto offer : myJobOffers) {
        unsigned int amountAble = moneyLeft / offer->wage;
        if (amountAble > offer->amountLeft) {
            offer->amountLeft = amountAble;
        }
        moneyLeft -= (offer->wage * offer->amountLeft);
    }
}


void Firm::accept_jobOffer_response(std::shared_ptr<JobOffer> jobOffer) {
    std::lock_guard<std::mutex> lock(myMutex);
    util::print_status(this, "Accepting jobOffer response...");
    money -= jobOffer->wage;
    laborHired += jobOffer->labor;
    jobOffer->amountLeft--;
    jobOffer->amountTaken++;
}
