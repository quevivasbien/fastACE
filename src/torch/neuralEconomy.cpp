#include "neuralEconomy.h"

namespace neural {

NeuralEconomy::NeuralEconomy(
    std::vector<std::string> goods,
    unsigned int maxAgents
) : Economy(goods), maxAgents(maxAgents) {}


std::shared_ptr<NeuralEconomy> NeuralEconomy::init(
    std::vector<std::string> goods,
    unsigned int maxAgents,
    std::shared_ptr<DecisionNetHandler> handler
) {
    std::shared_ptr<NeuralEconomy> self(
        new NeuralEconomy(goods, maxAgents)
    );
    self->handler = handler;
    self->handler->economy = self;
    return self;
}

std::shared_ptr<NeuralEconomy> NeuralEconomy::init(
    std::vector<std::string> goods,
    unsigned int maxAgents
) {
    std::shared_ptr<NeuralEconomy> self(
        new NeuralEconomy(goods, maxAgents)
    );
    self->handler = std::make_shared<DecisionNetHandler>(self);
    return self;
}


void NeuralEconomy::add_agent(std::shared_ptr<Person> person) {
    Economy::add_agent(person);
    update_agentMap(person);
}

void NeuralEconomy::add_agent(std::shared_ptr<Firm> firm) {
    Economy::add_agent(firm);
    update_agentMap(firm);
}

void NeuralEconomy::update_agentMap(std::shared_ptr<Agent> agent) {
    std::lock_guard<std::mutex> lock(mutex);
    agentMap[agent] = totalAgents++;

    if (totalAgents >= maxAgents) {
        std::cout <<
            "WARNING: You've hit the maximum number of agents allowed for this economy!"
                << std::endl;
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
