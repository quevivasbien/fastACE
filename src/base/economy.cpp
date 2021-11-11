#include "base.h"

Economy::Economy(std::vector<std::string> goods) : goods(goods), numGoods(goods.size()) {
    rng = get_rng();
}

std::shared_ptr<Person> Economy::add_person() {
    return create<Person>(this);
}

void Economy::add_agent(std::shared_ptr<Agent> agent) {
    std::lock_guard<std::mutex> lock(mutex);
    assert(agent->get_economy() == this);
    agents.push_back(agent);

    agentMap[agent] = totalAgents++;

    if (totalAgents >= maxAgents) {
        std::cout << "WARNING: You've hit the maximum number of agents allowed for this economy!" << std::endl;
    }
}

std::shared_ptr<Firm> Economy::add_firm(std::shared_ptr<Agent> firstOwner) {
    return create<Firm>(this, firstOwner);
}

const std::string* Economy::get_name_for_good_id(unsigned int id) const {
    return &goods[id];
}

const std::vector<std::shared_ptr<Agent>>& Economy::get_agents() const { return agents; }

unsigned int Economy::get_numAgents() const { return agents.size(); }

const std::vector<std::string>& Economy::get_goods() const { return goods; }

unsigned int Economy::get_numGoods() const { return numGoods; }

const std::vector<std::shared_ptr<const Offer>>& Economy::get_market() const { return market; }

const std::vector<std::shared_ptr<const JobOffer>>& Economy::get_jobMarket() const { return jobMarket; }

std::default_random_engine Economy::get_rng() const { return rng; }


void Economy::add_offer(std::shared_ptr<const Offer> offer) {
    std::lock_guard<std::mutex> lock(mutex);
    market.push_back(offer);
}
void Economy::add_jobOffer(std::shared_ptr<const JobOffer> jobOffer) {
    std::lock_guard<std::mutex> lock(mutex);
    jobMarket.push_back(jobOffer);
}

unsigned int Economy::get_id_for_agent(std::shared_ptr<Agent> agent) {
    return agentMap[agent];
}

unsigned int Economy::get_totalAgents() const {
    return totalAgents;
}

unsigned int Economy::get_maxAgents() const {
    return maxAgents;
}


void run_agents_(
    const std::vector<std::shared_ptr<Agent>>* const agents,
    unsigned int startIdx, unsigned int endIdx
) {
    for (unsigned int i = startIdx; i < endIdx; i++) {
        (*agents)[i]->time_step();
    }
}

void run_agents(const std::vector<std::shared_ptr<Agent>>* const agents) {
    // runs time_step for a vector of agents, multithreaded
    unsigned int numAgents = agents->size();
    auto numThreads = std::thread::hardware_concurrency();
    unsigned int agentsPerThread = numAgents / numThreads;
    unsigned int extras = numAgents % numThreads;
    std::vector<unsigned int> indices(numThreads + 1);
    indices[0] = 0;
    for (unsigned int i = 1; i <= numThreads; i++) {
        indices[i] = indices[i-1] + agentsPerThread + (i <= extras);
    }
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (unsigned int i = 0; i < numThreads; i++) {
        if (indices[i] != indices[i+1]) {
            threads.push_back(
                std::thread(
                    run_agents_,
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
    for (auto agent : agents) {
        if (agent->get_time() != time) {
            return false;
        }
    }
    time++;
    // now actually step
    // agents are shuffled first
    std::shuffle(std::begin(agents), std::end(agents), rng);
    if (constants::multithreaded) {
        run_agents(&agents);
    }
    else {
        for (auto agent : agents) {
            agent->time_step();
        }
    }
    flush<const Offer>(market);
    flush<const JobOffer>(jobMarket);
    return true;
}


void Economy::print_summary() const {
    std::cout << "\n----------\n"
        << "Memory ID: " << this << " (Economy)\n"
        << "----------\n";
    std::cout << "Time: " << time << "\n\n";
    std::cout << "Offers:\n";
    for (auto offer : market) {
        std::cout << "Offerer: " << offer->offerer << " ~ amt left: " << offer->amountLeft
            << " ~ amt taken: " << offer->amountTaken
            << "\n price: " << offer->price << " ~ quantitities " << offer->quantities.transpose()
            << '\n';
    }
    std::cout << "\nJob Offers:\n";
    for (auto offer : jobMarket) {
        std::cout << "Offerer: " << offer->offerer << " ~ amt left: " << offer->amountLeft
            << " ~ amt taken: " << offer->amountTaken
            << "\n wage: " << offer->wage << " ~ labor " << offer->labor
            << '\n';
    }
    std::cout << "\n\n";
}
