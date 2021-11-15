#ifndef NEURAL_ECONOMY_H
#define NEURAL_ECONOMY_H

#include <vector>
#include <string>
#include <unordered_map>
#include "base.h"

namespace neural {

class NeuralEconomy : public Economy {
    /**
    This is an economy that keeps track of all agents in an unordered map,
    and assigns each agent a unique int id;
    this is neceessary for use with neural net decision makers.
    maxAgents is the maximum number of unique Agents that can be created in this economy.
    totalAgents is the number of unique Agents that have been created so far.
    */
public:
    NeuralEconomy(std::vector<std::string> goods, unsigned int maxAgents);

    virtual void add_agent(std::shared_ptr<Person> person) override;
    virtual void add_agent(std::shared_ptr<Firm> firm) override;

    unsigned int get_id_for_agent(std::shared_ptr<Agent> agent);
    unsigned int get_totalAgents() const;
    unsigned int get_maxAgents() const;

protected:
    void update_agentMap(std::shared_ptr<Agent> agent);

    std::unordered_map<std::shared_ptr<Agent>, unsigned int> agentMap;
    unsigned int totalAgents = 0;
    unsigned int maxAgents;
};

} // namespace neural

#endif
