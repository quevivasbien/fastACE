#ifndef ADVANTAGE_ACTOR_CRITIC_H
#define ADVANTAGE_ACTOR_CRITIC_H

#include <torch/torch.h>
#include <string>
#include <thread>
#include <mutex>
#include <cmath>
#include "neuralEconomy.h"
#include "utilMaxer.h"
#include "profitMaxer.h"
#include "util.h"
#include "constants.h"
#include "neuralConstants.h"


namespace neural {


struct LRScheduler {
    // Performs a function similar to torch's ReduceLROnPlateau
    LRScheduler(
        torch::optim::Adam* optimizer,
        unsigned int episodeBatchSize,
        unsigned int patience,
        float decayMultiplier,
        std::string name
    );

    void decay_lr();
    void update_lr(float loss);

    torch::optim::Adam* optimizer;
    std::vector<float> lossHistory;
    unsigned int episodeBatchSize;
    unsigned int patience;
    float decayMultiplier;

    float bestBatchLoss = HUGE_VALF;
    unsigned int numBadBatches = 0;

    std::string name;
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
        unsigned int episodeBatchSizeForLRDecay,
        unsigned int patienceForLRDecay,
        float multiplierForLRDecay
    );

    AdvantageActorCritic(
        std::shared_ptr<DecisionNetHandler> handler,
        float initialLR
    );

    AdvantageActorCritic(
        std::shared_ptr<DecisionNetHandler> handler
    );

    std::shared_ptr<DecisionNetHandler> handler;

    Adam purchaseNetOptim;
    Adam firmPurchaseNetOptim;
    Adam laborSearchNetOptim;
    Adam consumptionNetOptim;
    Adam productionNetOptim;
    Adam offerNetOptim;
    Adam jobOfferNetOptim;
    Adam valueNetOptim;
    Adam firmValueNetOptim;

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
            std::unordered_map<Agent*, torch::Tensor>
        >& logProbas,
        Adam& optimizer,
        Agent* agent,
        const torch::Tensor& advantage
    );

    // pair.first = loss, pair.second = advantage vector
    std::pair<torch::Tensor, torch::Tensor> get_advantage_for_person(Person* person);

    float get_loss_for_person_in_episode(std::weak_ptr<Person> person, unsigned int numTotalPersons);

    // pair.first = loss, pair.second = advantage vector
    std::pair<torch::Tensor, torch::Tensor> get_advantage_for_firm(Firm* firm);

    float get_loss_for_firm_in_episode(std::weak_ptr<Firm> firm, unsigned int numTotalFirms);

    void get_loss_for_persons_multithreaded_(
        const std::vector<std::weak_ptr<Person>>& persons,
        unsigned int startIdx,
        unsigned int endIdx,
        float* loss
    );
    float get_loss_for_persons_multithreaded();
    void get_loss_for_firms_multithreaded_(
        const std::vector<std::weak_ptr<Firm>>& firms,
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
