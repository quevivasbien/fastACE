#include <vector>
#include <memory>
#include "base.h"

Firm::Firm(Economy* economy, std::shared_ptr<Agent> owner)
    : Agent(economy) {
        owners.push_back(owner);
    }

Firm::Firm(Economy* economy, std::vector<std::shared_ptr<Agent>> owners, Eigen::ArrayXd inventory, double money)
    : Agent(economy, inventory, money), owners(owners) {}


double Firm::get_laborHired() const {
    return laborHired;
}


bool Firm::time_step() {
    if (!Agent::time_step()) {
        return false;
    }
    else {
        check_myJobOffers();
        flush_myJobOffers();
        search_for_laborers();
        buy_goods();
        produce();
        sell_goods();
        laborHired = 0.0;
        return true;
    }
}

void Firm::post_jobOffer(std::shared_ptr<JobOffer> jobOffer) {
    assert(jobOffer->offerer == shared_from_this());
    economy->add_jobOffer(jobOffer);
    myJobOffers.push_back(jobOffer);
}


bool Firm::review_jobOffer_response(
    std::shared_ptr<Person> responder,
    std::shared_ptr<const JobOffer> jobOffer
) {
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
    // all good, let's go!
    accept_jobOffer_response(myCopy);
    return true;
}


void Firm::check_myJobOffers() {
    double moneyLeft = money;
    for (auto offer : myJobOffers) {
        unsigned int amountAble = moneyLeft / offer->wage;
        if (amountAble > offer->amountLeft) {
            offer->amountLeft = amountAble;
        }
    }
}


void Firm::accept_jobOffer_response(std::shared_ptr<JobOffer> jobOffer) {
    money -= jobOffer->wage;
    labor += jobOffer->labor;
    jobOffer->amountLeft--;
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
