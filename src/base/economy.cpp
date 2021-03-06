#include "base.h"

Economy::Economy(std::vector<std::string> goods) : goods(goods), numGoods(goods.size()) {
    rng = util::get_rng();
}

std::shared_ptr<Person> Economy::add_person() {
    return util::create<Person>(this);
}

void Economy::add_agent(std::shared_ptr<Person> person) {
    std::lock_guard<std::mutex> lock(mutex);
    assert(person->get_economy() == this);
    persons.push_back(person);
    persons_weak.push_back(std::weak_ptr<Person>(person));
}

std::shared_ptr<Firm> Economy::add_firm() {
    return util::create<Firm>(this);
}

void Economy::add_agent(std::shared_ptr<Firm> firm) {
    std::lock_guard<std::mutex> lock(mutex);
    assert(firm->get_economy() == this);
    firms.push_back(firm);
    firms_weak.push_back(std::weak_ptr<Firm>(firm));
}

const std::string& Economy::get_name_for_good_id(unsigned int id) const {
    return goods[id];
}

const std::vector<std::weak_ptr<Person>>& Economy::get_persons() const {
    return persons_weak;
}

const std::vector<std::weak_ptr<Firm>>& Economy::get_firms() const { 
    return firms_weak;
}

const std::vector<std::string>& Economy::get_goods() const { return goods; }

unsigned int Economy::get_numGoods() const { return numGoods; }

const std::vector<std::weak_ptr<const Offer>>& Economy::get_market() const { return market; }

const std::vector<std::weak_ptr<const JobOffer>>& Economy::get_jobMarket() const { return jobMarket; }

std::default_random_engine Economy::get_rng() const { return rng; }


void Economy::add_offer(std::weak_ptr<const Offer> offer) {
    std::lock_guard<std::mutex> lock(mutex);
    market.push_back(offer);
}
void Economy::add_jobOffer(std::weak_ptr<const JobOffer> jobOffer) {
    std::lock_guard<std::mutex> lock(mutex);
    jobMarket.push_back(jobOffer);
}


template <typename A>
void run_agents_(
    const std::vector<std::shared_ptr<A>>* const agents,
    unsigned int startIdx, unsigned int endIdx
) {
    for (unsigned int i = startIdx; i < endIdx; i++) {
        (*agents)[i]->time_step();
    }
}

template <typename A>
void run_agents(const std::vector<std::shared_ptr<A>>* const agents) {
    // runs time_step for a vector of agents, multithreaded
    std::vector<unsigned int> indices = util::get_indices_for_multithreading(agents->size());
    std::vector<std::thread> threads;
    threads.reserve(constants::numThreads);
    for (unsigned int i = 0; i < constants::numThreads; i++) {
        if (indices[i] != indices[i+1]) {
            threads.push_back(
                std::thread(
                    run_agents_<A>,
                    agents,
                    indices[i],
                    indices[i+1]
                )
            );
        }
    }
    for (unsigned int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }
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
    time++;
    // now actually step
    // agents are shuffled first
    std::shuffle(std::begin(persons), std::end(persons), rng);
    std::shuffle(std::begin(firms), std::end(firms), rng);
    // persons go first, then firms
    if (constants::multithreaded) {
        run_agents(&persons);
        run_agents(&firms);
    }
    else {
        for (auto person : persons) {
            person->time_step();
        }
        for (auto firm : firms) {
            firm->time_step();
        }
    }
    util::flush(market);
    util::flush(jobMarket);
    if (constants::verbose >= 3) {
        print_summary();
    }
    if (constants::verbose >= 4) {
        for (auto person : persons) {
            person->print_summary();
        }
        for (auto firm : firms) {
            firm->print_summary();
        }
    }
    return true;
}

unsigned int Economy::get_time() const {
    return time;
}


std::string Economy::get_typename() const {
    return "Economy";
}


void Economy::print_summary() const {
    std::cout << "\n----------\n"
        << "Memory ID: " << this << " (" << get_typename() << ")\n"
        << "----------\n";
    std::cout << "Time: " << time << "\n\n";
    std::cout << "Offers:\n";
    for (auto offer_ : market) {
        auto offer = offer_.lock();
        if (offer == nullptr) { continue; }
        std::cout << "Offerer: " << offer->offerer.lock() << " ~ amt left: " << offer->amountLeft
            << " ~ amt taken: " << offer->amountTaken
            << "\n price: " << offer->price << " ~ quantitities " << offer->quantities.transpose()
            << '\n';
    }
    std::cout << "\nJob Offers:\n";
    for (auto offer_ : jobMarket) {
        auto offer = offer_.lock();
        if (offer == nullptr) { continue; }
        std::cout << "Offerer: " << offer->offerer.lock() << " ~ amt left: " << offer->amountLeft
            << " ~ amt taken: " << offer->amountTaken
            << "\n wage: " << offer->wage << " ~ labor " << offer->labor
            << '\n';
    }
    std::cout << "\n\n";
}
