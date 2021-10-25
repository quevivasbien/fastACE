#include <vector>
#include <memory>
#include "base.h"

Firm::Firm(Economy* economy, std::shared_ptr<Agent> owner)
    : Agent(economy) {
        owners.push_back(owner);
    }

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
        print_status(this, "Checking my job offers...");
        check_myJobOffers();
        myMutex.lock();
        flush<JobOffer>(myJobOffers);
        myMutex.unlock();
        print_status(this, "Buying goods...");
        buy_goods();
        print_status(this, "Producing...");
        produce();
        print_status(this, "Selling goods...");
        sell_goods();
        pay_dividends();
        laborHired = 0.0;
        print_status(this, "Searching for laborers...");
        search_for_laborers();
        return true;
    }
}

void Firm::post_jobOffer(std::shared_ptr<JobOffer> jobOffer) {
    std::lock_guard<std::mutex> lock(myMutex);
    assert(jobOffer->offerer == shared_from_this());
    economy->add_jobOffer(jobOffer);
    myJobOffers.push_back(jobOffer);
}


bool Firm::review_jobOffer_response(
    std::shared_ptr<Person> responder,
    std::shared_ptr<const JobOffer> jobOffer
) {
    myMutex.lock();
    // check that the offer is in myOffers
    std::shared_ptr<JobOffer> myCopy;
    for (auto myOffer : myJobOffers) {
        if (myOffer == jobOffer) {
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
        // mark for removal
        myCopy->amountLeft = 0;
        return false;
    }
    myMutex.unlock();
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
    }
}


void Firm::accept_jobOffer_response(std::shared_ptr<JobOffer> jobOffer) {
    std::lock_guard<std::mutex> lock(myMutex);
    money -= jobOffer->wage;
    labor += jobOffer->labor;
    jobOffer->amountLeft--;
    jobOffer->amountTaken++;
}
