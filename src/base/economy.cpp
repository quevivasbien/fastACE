#include "base.h"

Economy::Economy(std::vector<std::string> goods) : goods(goods), numGoods(goods.size()) {
    rng = get_rng();
}

std::shared_ptr<Person> Economy::add_person() {
    return create<Person>(this);
}

void Economy::add_agent(std::shared_ptr<Person> person) {
    assert(person->get_economy() == this);
    persons.push_back(person);
}

std::shared_ptr<Firm> Economy::add_firm(std::shared_ptr<Agent> firstOwner) {
    return create<Firm>(this, firstOwner);
}

void Economy::add_agent(std::shared_ptr<Firm> firm) {
    assert(firm->get_economy() == this);
    firms.push_back(firm);
}

const std::string* Economy::get_name_for_good_id(unsigned int id) const {
    return &goods[id];
}

const std::vector<std::shared_ptr<Person>>& Economy::get_persons() const { return persons; }
const std::vector<std::shared_ptr<Firm>>& Economy::get_firms() const { return firms; }
const std::vector<std::string>& Economy::get_goods() const { return goods; }
unsigned int Economy::get_numGoods() const { return numGoods; }
const std::vector<std::shared_ptr<const Offer>>& Economy::get_market() const { return market; }
const std::vector<std::shared_ptr<const JobOffer>>& Economy::get_jobMarket() const { return laborMarket; }
std::default_random_engine Economy::get_rng() const { return rng; }

void Economy::add_offer(std::shared_ptr<const Offer> offer) { market.push_back(offer); }
void Economy::add_jobOffer(std::shared_ptr<const JobOffer> jobOffer) { laborMarket.push_back(jobOffer); }


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
    time++;
    // TODO: randomize order of movement
    // now actually step
    for (auto person : persons) {
        person->time_step();
    }
    for (auto firm : firms) {
        firm->time_step();
    }
    flush<const Offer>(market);
    flush<const JobOffer>(laborMarket);
    return true;
}
