#ifndef ADVANTAGE_ACTOR_CRITIC_H
#define ADVANTAGE_ACTOR_CRITIC_H

#include <torch/torch.h>
#include <string>
#include "neuralEconomy.h"
#include "utilMaxer.h"
#include "profitMaxer.h"
#include "util.h"


namespace neural {

const double DEFAULT_LEARNING_RATE = 0.01;


struct AdvantageActorCritic {
    using Adam = torch::optim::Adam;

    AdvantageActorCritic(
        std::shared_ptr<DecisionNetHandler> handler,
        double purchaseNetLR,
        double firmPurchaseNetLR,
        double laborSearchNetLR,
        double consumptionNetLR,
        double productionNetLR,
        double offerNetLR,
        double jobOfferNetLR,
        double valueNetLR,
        double firmValueNetLR
    ) : handler(handler),
        purchaseNetOptim(
            std::make_shared<Adam>(
                handler->purchaseNet->parameters(),
                purchaseNetLR
            )
        ),
        firmPurchaseNetOptim(
            std::make_shared<Adam>(
                handler->firmPurchaseNet->parameters(),
                firmPurchaseNetLR
            )
        ),
        laborSearchNetOptim(
            std::make_shared<Adam>(
                handler->firmPurchaseNet->parameters(),
                firmPurchaseNetLR
            )
        ),
        consumptionNetOptim(
            std::make_shared<Adam>(
                handler->laborSearchNet->parameters(),
                laborSearchNetLR
            )
        ),
        productionNetOptim(
            std::make_shared<Adam>(
                handler->consumptionNet->parameters(),
                consumptionNetLR
            )
        ),
        offerNetOptim(
            std::make_shared<Adam>(
                handler->offerNet->parameters(),
                offerNetLR
            )
        ),
        jobOfferNetOptim(
            std::make_shared<Adam>(
                handler->jobOfferNet->parameters(),
                jobOfferNetLR
            )
        ),
        valueNetOptim(
            std::make_shared<Adam>(
                handler->valueNet->parameters(),
                valueNetLR
            )
        ),
        firmValueNetOptim(
            std::make_shared<Adam>(
                handler->firmValueNet->parameters(),
                firmValueNetLR
            )
        )
    {}

    AdvantageActorCritic(
        std::shared_ptr<DecisionNetHandler> handler
    ) : AdvantageActorCritic(
        handler,
        DEFAULT_LEARNING_RATE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_LEARNING_RATE
    ) {}

    std::shared_ptr<DecisionNetHandler> handler;

    std::shared_ptr<Adam> purchaseNetOptim;
    std::shared_ptr<Adam> firmPurchaseNetOptim;
    std::shared_ptr<Adam> laborSearchNetOptim;
    std::shared_ptr<Adam> consumptionNetOptim;
    std::shared_ptr<Adam> productionNetOptim;
    std::shared_ptr<Adam> offerNetOptim;
    std::shared_ptr<Adam> jobOfferNetOptim;
    std::shared_ptr<Adam> valueNetOptim;
    std::shared_ptr<Adam> firmValueNetOptim;

    void backprop_on(
        const std::vector<
            std::unordered_map<std::shared_ptr<Agent>, torch::Tensor>
        >& logProbas,
        std::shared_ptr<Adam> optimizer,
        std::shared_ptr<Agent> agent,
        const torch::Tensor& advantage
    ) {
        auto loss = torch::tensor(0.0, torch::requires_grad(true));
        for (int t = 0; t < handler->time; t++) {
            auto logProbaSearch = logProbas[t].find(agent);
            if (logProbaSearch != logProbas[t].end()) {
                auto logProba = logProbaSearch->second;
                loss = loss + logProba * advantage[t];
            }
            else {
                print_status(
                    agent, "WARNING: Can't find in log probas at time " + std::to_string(t)
                );
            }
        }
        (loss / handler->time).backward({}, true);
    }

    void backprop_on_person_in_episode(std::shared_ptr<Person> person) {
        auto person_as_utilmaxer = std::static_pointer_cast<UtilMaxer>(person);

        // Calculate advantage time series
        // advantage is realized utility minus predicted value in each state
        // Then critic loss is sum of squared advantage
        auto critic_loss = torch::tensor(0.0, torch::requires_grad(true));
        auto advantage = torch::empty(handler->time);
        auto q = torch::tensor(0.0, torch::requires_grad(true));
        for (int t = handler->time - 1; t >= 0; t--) {
            auto reward_search = handler->rewards[t].find(person);
            auto value_search = handler->values[t].find(person);
            if (
                (reward_search != handler->rewards[t].end())
                && (value_search != handler->values[t].end())
            ) {
                auto reward = reward_search->second;
                auto value = value_search->second;
                q = reward + person_as_utilmaxer->get_discountRate() * q;
                auto advantage_t = q - value.squeeze();
                advantage[t] = advantage_t.detach();
                critic_loss = critic_loss + advantage_t.pow(2);
            }
            else {
                print_status(
                    person_as_utilmaxer,
                    "WARNING: Can't find in rewards || value map at time " + std::to_string(t)
                );
            }
        }

        // Backpropagate on critic (value function) loss
        // Divide by time to keep loss normalized
        (critic_loss / handler->time).backward({}, true);

        // Backpropagate on the other person decision nets
        backprop_on(
            handler->purchaseNetLogProba,
            purchaseNetOptim,
            person,
            advantage
        );
        backprop_on(
            handler->laborSearchNetLogProba,
            laborSearchNetOptim,
            person,
            advantage
        );
        backprop_on(
            handler->consumptionNetLogProba,
            consumptionNetOptim,
            person,
            advantage
        );
    }

    void backprop_on_firm_in_episode(std::shared_ptr<Firm> firm) {
        // here advantage is realized profit minus predicted value in each state
        auto critic_loss = torch::tensor(0.0, torch::requires_grad(true));
        auto advantage = torch::empty(handler->time);
        auto q = torch::tensor(0.0, torch::requires_grad(true));
        for (int t = handler->time - 1; t >= 0; t--) {
            auto reward_search = handler->rewards[t].find(firm);
            auto value_search = handler->values[t].find(firm);
            if (
                (reward_search != handler->rewards[t].end())
                && (value_search != handler->values[t].end())
            ) {
                auto reward = reward_search->second;
                auto value = value_search->second;
                q = reward + q;  // no discounting with firms
                auto advantage_t = q - value.squeeze();
                advantage[t] = advantage_t.detach();
                critic_loss = critic_loss + advantage_t.pow(2);
            }
        }

        // Backprop on value function loss
        critic_loss.backward();

        // Backprop on other firm decision nets
        backprop_on(
            handler->firmPurchaseNetLogProba,
            firmPurchaseNetOptim,
            firm,
            advantage
        );
        backprop_on(
            handler->productionNetLogProba,
            productionNetOptim,
            firm,
            advantage
        );
        backprop_on(
            handler->offerNetLogProba,
            offerNetOptim,
            firm,
            advantage
        );
        backprop_on(
            handler->jobOfferNetLogProba,
            jobOfferNetOptim,
            firm,
            advantage
        );
    }

    void zero_all_grads() {
        purchaseNetOptim->zero_grad();
        firmPurchaseNetOptim->zero_grad();
        laborSearchNetOptim->zero_grad();
        consumptionNetOptim->zero_grad();
        productionNetOptim->zero_grad();
        offerNetOptim->zero_grad();
        jobOfferNetOptim->zero_grad();
        valueNetOptim->zero_grad();
        firmValueNetOptim->zero_grad();
    }

    void all_optims_step() {
        purchaseNetOptim->step();
        firmPurchaseNetOptim->step();
        laborSearchNetOptim->step();
        consumptionNetOptim->step();
        productionNetOptim->step();
        offerNetOptim->step();
        jobOfferNetOptim->step();
        valueNetOptim->step();
        firmValueNetOptim->step();
    }

    void train_on_episode() {
        zero_all_grads();
        for (auto person : handler->economy->get_persons()) {
            backprop_on_person_in_episode(person);
        }
        for (auto firm : handler->economy->get_firms()) {
            backprop_on_firm_in_episode(firm);
        }
        all_optims_step();
    }
 
};


}

#endif
