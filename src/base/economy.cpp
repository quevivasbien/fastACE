#include "base.h"

Economy::Economy(std::vector<std::string> goods) : goods(goods), numGoods(goods.size()) {}

void Economy::add_person() {
    std::make_shared<Person>(this);
}

void Economy::add_person(std::shared_ptr<Person> person) {
    persons.push_back(person);
}

void Economy::add_firm(std::shared_ptr<Agent> firstOwner) {
    std::make_shared<Firm>(this, firstOwner);
}

void Economy::add_firm(std::shared_ptr<Firm> firm) {
    firms.push_back(firm);
}

const std::string* Economy::get_name_for_good_id(unsigned int id) const {
    return &goods[id];
}

const std::vector<std::shared_ptr<Person>>& Economy::get_persons() const { return persons; }
const std::vector<std::shared_ptr<Firm>>& Economy::get_firms() const { return firms; }
const std::vector<std::string>& Economy::get_goods() const { return goods; }
unsigned int Economy::get_numGoods() const { return numGoods; }
const std::vector<std::shared_ptr<Offer>>& Economy::get_market() const { return market; }
const std::vector<std::shared_ptr<JobOffer>>& Economy::get_laborMarket() const { return laborMarket; }

void Economy::add_offer(std::shared_ptr<Offer> offer) { market.push_back(offer); }
void Economy::add_jobOffer(std::shared_ptr<JobOffer> jobOffer) { laborMarket.push_back(jobOffer); }

void Economy::flush_market() {
    flush_offers<Offer>(market);
}

void Economy::flush_labor_market() {
    flush_offers<JobOffer>(laborMarket);
}

bool Economy::time_step() {
    // check that all agents have caught up before stepping
    for (auto person : persons) {
        if (person->get_time() != time) {
            return false;
        }
    }
    for (auto firm : firms) {
        if (firm->get_time() != time) {
            return false;
        }
    }
    // now actually step
    for (auto person : persons) {
        person->time_step();
    }
    for (auto firm : firms) {
        firm->time_step();
    }
    flush_market();
    flush_labor_market();
    return true;
}
