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


class LRScheduler {
public:
    // Performs a function similar to torch's ReduceLROnPlateau
    LRScheduler(
        torch::optim::Adam* optimizer,
        unsigned int episodeBatchSize,
        unsigned int patience,
        double decayMultiplier,
        unsigned int cosinePeriod,
        std::string name
    );

    void scale_lr(double multiplier);
    void update_lr(double loss);

    torch::optim::Adam* optimizer;
    std::vector<double> lossHistory;
    unsigned int episodeBatchSize;
    unsigned int patience;
    double decayMultiplier;
    unsigned int cosinePeriod;

    double bestBatchLoss = HUGE_VALF;
    unsigned int numBadBatches = 0;

    unsigned int cosineTimer = 0;

    std::string name;
};


class AdvantageActorCritic {
    using Adam = torch::optim::Adam;
    
public:
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
        double firmValueNetLR,
        unsigned int episodeBatchSizeForLRDecay,
        unsigned int patienceForLRDecay,
        double multiplierForLRDecay,
        unsigned int cosinePeriod
    );

    AdvantageActorCritic(
        std::shared_ptr<DecisionNetHandler> handler,
        double initialLR
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
    double purchaseNetLoss = 0.0;
    double firmPurchaseNetLoss = 0.0;
    double laborSearchNetLoss = 0.0;
    double consumptionNetLoss = 0.0;
    double productionNetLoss = 0.0;
    double offerNetLoss = 0.0;
    double jobOfferNetLoss = 0.0;
    double valueNetLoss = 0.0;
    double firmValueNetLoss = 0.0;

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

    double get_loss_for_person_in_episode(std::weak_ptr<Person> person, unsigned int numTotalPersons);

    // pair.first = loss, pair.second = advantage vector
    std::pair<torch::Tensor, torch::Tensor> get_advantage_for_firm(Firm* firm);

    double get_loss_for_firm_in_episode(std::weak_ptr<Firm> firm, unsigned int numTotalFirms);

    void get_loss_for_persons_multithreaded_(
        const std::vector<std::weak_ptr<Person>>& persons,
        unsigned int startIdx,
        unsigned int endIdx,
        double* loss
    );
    double get_loss_for_persons_multithreaded();
    void get_loss_for_firms_multithreaded_(
        const std::vector<std::weak_ptr<Firm>>& firms,
        unsigned int startIdx,
        unsigned int endIdx,
        double* loss
    );
    double get_loss_for_firms_multithreaded();

    double get_loss_multithreaded();

    void zero_all_grads();

    void zero_all_tracked_losses();

    void update_lr_schedulers();

    void all_optims_step();

    double train_on_episode();
 
};


} // namespace neural

#endif
