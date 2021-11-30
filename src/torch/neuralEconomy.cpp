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

    if (totalAgents > maxAgents) {
        pprint_status(
            1,
            this,
            "WARNING: You've EXCEEDED the maximum number of agents allowed for this economy!"
        );
    }
    else if (totalAgents == maxAgents) {
        pprint_status(
            2,
            this,
            "WARNING: You've hit the maximum number of agents allowed for this economy!"
        );
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

std::string NeuralEconomy::get_typename() const {
    return "NeuralEconomy";
}

} // namespace neural
