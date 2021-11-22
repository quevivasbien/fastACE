#ifndef ADVANTAGE_ACTOR_CRITIC_H
#define ADVANTAGE_ACTOR_CRITIC_H

#include <torch/torch.h>
#include <string>
#include <thread>
#include <mutex>
#include "neuralEconomy.h"
#include "utilMaxer.h"
#include "profitMaxer.h"
#include "util.h"
#include "constants.h"


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
    );

    AdvantageActorCritic(
        std::shared_ptr<DecisionNetHandler> handler
    );

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

    std::mutex myMutex;

    torch::Tensor get_loss_from_logProba(
        const std::vector<
            std::unordered_map<std::shared_ptr<Agent>, torch::Tensor>
        >& logProbas,
        std::shared_ptr<Adam> optimizer,
        std::shared_ptr<Agent> agent,
        const torch::Tensor& advantage
    );

    torch::Tensor get_loss_for_person_in_episode(std::shared_ptr<Person> person);

    torch::Tensor get_loss_for_firm_in_episode(std::shared_ptr<Firm> firm);

    void get_loss_for_persons_multithreaded_(
        const std::vector<std::shared_ptr<Person>>& persons,
        unsigned int startIdx,
        unsigned int endIdx,
        torch::Tensor* loss
    );
    torch::Tensor get_loss_for_persons_multithreaded();
    void get_loss_for_firms_multithreaded_(
        const std::vector<std::shared_ptr<Firm>>& firms,
        unsigned int startIdx,
        unsigned int endIdx,
        torch::Tensor* loss
    );
    torch::Tensor get_loss_for_firms_multithreaded();

    void zero_all_grads();

    void all_optims_step();

    float train_on_episode();
 
};


} // namespace neural

#endif
