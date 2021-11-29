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

const float DEFAULT_LEARNING_RATE = 0.001;
const float DEFAULT_LR_SCHEDULE_THRESHOLD = 0.1;
const unsigned int DEFAULT_LR_SCHEDULE_BATCH_INTERVAL = 40;
const float DEFAULT_LR_SCHEDULE_DECAY_MULTIPLIER = 0.5;


struct LRScheduler {
    LRScheduler(
        std::shared_ptr<torch::optim::Adam> optimizer,
        float threshold,
        unsigned int episodeGroupSize,
        float decayMultiplier
    );

    void decay_lr();
    void update_lr(float loss);

    std::shared_ptr<torch::optim::Adam> optimizer;
    std::vector<float> lossHistory;
    float threshold;
    float decayMultiplier;
    unsigned int episodeGroupSize;
};


struct AdvantageActorCritic {
    using Adam = torch::optim::Adam;

    AdvantageActorCritic(
        std::shared_ptr<DecisionNetHandler> handler,
        float purchaseNetLR,
        float firmPurchaseNetLR,
        float laborSearchNetLR,
        float consumptionNetLR,
        float productionNetLR,
        float offerNetLR,
        float jobOfferNetLR,
        float valueNetLR,
        float firmValueNetLR,
        float scheduleThreshold,
        unsigned int scheduleBatchInterval,
        float scheduleDecayMultipler
    );

    AdvantageActorCritic(
        std::shared_ptr<DecisionNetHandler> handler,
        float initialLR
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

    // these are for keeping track of where the loss is coming from
    // the idea is to track these to adjust learning rates as needed
    float purchaseNetLoss = 0.0;
    float firmPurchaseNetLoss = 0.0;
    float laborSearchNetLoss = 0.0;
    float consumptionNetLoss = 0.0;
    float productionNetLoss = 0.0;
    float offerNetLoss = 0.0;
    float jobOfferNetLoss = 0.0;
    float valueNetLoss = 0.0;
    float firmValueNetLoss = 0.0;

    LRScheduler purchaseNetScheduler;
    LRScheduler firmPurchaseNetScheduler;
    LRScheduler laborSearchNetScheduler;
    LRScheduler consumptionNetScheduler;
    LRScheduler productionNetScheduler;
    LRScheduler offerNetScheduler;
    LRScheduler jobOfferNetScheduler;
    LRScheduler valueNetScheduler;
    LRScheduler firmValueNetScheduler;

    std::mutex myMutex;
    

    torch::Tensor get_loss_from_logProba(
        const std::vector<
            std::unordered_map<std::shared_ptr<Agent>, torch::Tensor>
        >& logProbas,
        std::shared_ptr<Adam> optimizer,
        std::shared_ptr<Agent> agent,
        const torch::Tensor& advantage
    );

    template <typename A>
    void get_loss_from_logProba_multithreaded(
        unsigned int numAgents,
        const std::vector<std::unordered_map<std::shared_ptr<Agent>, torch::Tensor>>& logProbas,
        std::shared_ptr<torch::optim::Adam> optimizer,
        const std::vector<std::shared_ptr<A>>& agents,
        const std::vector<torch::Tensor>& advantages,
        float* loss_
    ) {
        auto loss = torch::tensor(0.0);
        for (unsigned int i = 0; i < numAgents; i++) {
            loss = loss + get_loss_from_logProba(
                logProbas,
                optimizer,
                agents[i],
                advantages[i]
            );
        }
        loss = loss / static_cast<long>(numAgents * handler->time);
        loss.backward({}, true);
        {
            std::lock_guard<std::mutex> lock(myMutex);
            *loss_ = *loss_ + loss.item<float>();
        }
    }

    // pair.first = loss, pair.second = advantage vector
    std::pair<torch::Tensor, torch::Tensor> get_advantage_for_person(std::shared_ptr<Person> person);

    float get_loss_for_person_in_episode(std::shared_ptr<Person> person, unsigned int numTotalPersons);

    // pair.first = loss, pair.second = advantage vector
    std::pair<torch::Tensor, torch::Tensor> get_advantage_for_firm(std::shared_ptr<Firm> firm);

    float get_loss_for_firm_in_episode(std::shared_ptr<Firm> firm, unsigned int numTotalFirms);

    void get_loss_for_persons_multithreaded_(
        const std::vector<std::shared_ptr<Person>>& persons,
        unsigned int startIdx,
        unsigned int endIdx,
        float* loss
    );
    float get_loss_for_persons_multithreaded();
    void get_loss_for_firms_multithreaded_(
        const std::vector<std::shared_ptr<Firm>>& firms,
        unsigned int startIdx,
        unsigned int endIdx,
        float* loss
    );
    float get_loss_for_firms_multithreaded();

    float get_loss_multithreaded();

    void zero_all_grads();

    void zero_all_tracked_losses();

    void update_lr_schedulers();

    void all_optims_step();

    float train_on_episode();
 
};


} // namespace neural

#endif
