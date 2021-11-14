#include "neuralEconomy.h"

namespace neural {

NeuralEconomy::NeuralEconomy(
    std::vector<std::string> goods,
    unsigned int maxAgents
) : Economy(goods), maxAgents(maxAgents) {}

void NeuralEconomy::add_agent(std::shared_ptr<Agent> agent) {
    std::lock_guard<std::mutex> lock(mutex);
    assert(agent->get_economy() == this);
    agents.push_back(agent);

    agentMap[agent] = totalAgents++;

    if (totalAgents >= maxAgents) {
        std::cout << "WARNING: You've hit the maximum number of agents allowed for this economy!" << std::endl;
    }
}


unsigned int NeuralEconomy::get_id_for_agent(std::shared_ptr<Agent> agent) {
    return agentMap[agent];
}

unsigned int NeuralEconomy::get_totalAgents() const {
    return totalAgents;
}

unsigned int NeuralEconomy::get_maxAgents() const {
    return maxAgents;
}

} // namespace neural
