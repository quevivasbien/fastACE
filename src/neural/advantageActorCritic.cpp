#include "advantageActorCritic.h"


namespace neural {



LRScheduler::LRScheduler(
    torch::optim::Adam* optimizer,
    unsigned int episodeBatchSize,
    unsigned int patience,
    double decayMultiplier,
    unsigned int cosinePeriod,
    std::string name
) : optimizer(optimizer),
    episodeBatchSize(episodeBatchSize),
    patience(patience),
    decayMultiplier(decayMultiplier),
    cosinePeriod(cosinePeriod),
    name(name)
{}

double LRScheduler::get_lr() const {
    for (auto& group : optimizer->param_groups()) {
        if (group.has_options()) {
            return (static_cast<torch::optim::AdamOptions&>(group.options())).get_lr();
        }
    }
    // if we arrive here, that's a problem!!
    assert(false);
}

void LRScheduler::scale_lr(double multiplier) {
    for (auto& group : optimizer->param_groups()) {
        if (group.has_options()) {
            auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
            double newLR = options.get_lr() * multiplier;
            util::pprint(1, "Setting LR for " + name + " to " + util::format_sci_notation(newLR));
            options.set_lr(newLR);
        }
    }
}

void LRScheduler::update_lr(double loss) {
    lossHistory.push_back(loss);

    if (lossHistory.size() == episodeBatchSize) {
        double recentBatchLoss = 0.0;
        for (auto x : lossHistory) {
            recentBatchLoss += x;
        }
        if (recentBatchLoss < bestBatchLoss) {
            bestBatchLoss = recentBatchLoss;
            numBadBatches = 0;
        }
        else {
            numBadBatches++;
        }

        if (numBadBatches >= patience) {
            scale_lr(decayMultiplier);
            numBadBatches = 0;
        }

        lossHistory.clear();
    }

    if (++cosineTimer == cosinePeriod * episodeBatchSize * patience) {
        scale_lr(1.0 / decayMultiplier);
        cosineTimer = 0;
    }
}


AdvantageActorCritic::AdvantageActorCritic(
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
) : handler(handler),
    purchaseNetOptim(
        Adam(
            handler->purchaseNet->parameters(),
            purchaseNetLR
        )
    ),
    firmPurchaseNetOptim(
        Adam(
            handler->firmPurchaseNet->parameters(),
            firmPurchaseNetLR
        )
    ),
    laborSearchNetOptim(
        Adam(
            handler->firmPurchaseNet->parameters(),
            firmPurchaseNetLR
        )
    ),
    consumptionNetOptim(
        Adam(
            handler->laborSearchNet->parameters(),
            laborSearchNetLR
        )
    ),
    productionNetOptim(
        Adam(
            handler->consumptionNet->parameters(),
            consumptionNetLR
        )
    ),
    offerNetOptim(
        Adam(
            handler->offerNet->parameters(),
            offerNetLR
        )
    ),
    jobOfferNetOptim(
        Adam(
            handler->jobOfferNet->parameters(),
            jobOfferNetLR
        )
    ),
    valueNetOptim(
        Adam(
            handler->valueNet->parameters(),
            valueNetLR
        )
    ),
    firmValueNetOptim(
        Adam(
            handler->firmValueNet->parameters(),
            firmValueNetLR
        )
    ),

    purchaseNetScheduler(LRScheduler(
        &purchaseNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        cosinePeriod,
        "purchaseNet"
    )),
    firmPurchaseNetScheduler(LRScheduler(
        &firmPurchaseNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        cosinePeriod,
        "firmPurchaseNet"
    )),
    laborSearchNetScheduler(LRScheduler(
        &laborSearchNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        cosinePeriod,
        "laborSearchNet"
    )),
    consumptionNetScheduler(LRScheduler(
        &consumptionNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        cosinePeriod,
        "consumptionNet"
    )),
    productionNetScheduler(LRScheduler(
        &productionNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        cosinePeriod,
        "productionNet"
    )),
    offerNetScheduler(LRScheduler(
        &offerNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        cosinePeriod,
        "offerNet"
    )),
    jobOfferNetScheduler(LRScheduler(
        &jobOfferNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        cosinePeriod,
        "jobOfferNet"
    )),
    valueNetScheduler(LRScheduler(
        &valueNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        cosinePeriod,
        "valueNet"
    )),
    firmValueNetScheduler(LRScheduler(
        &firmValueNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        cosinePeriod,
        "firmValueNet"
    ))
{}

AdvantageActorCritic::AdvantageActorCritic(
    std::shared_ptr<DecisionNetHandler> handler,
    double initialLR
) : AdvantageActorCritic(
    handler,
    initialLR,
    initialLR,
    initialLR,
    initialLR,
    initialLR,
    initialLR,
    initialLR,
    initialLR,
    initialLR,
    DEFAULT_EPISODE_BATCH_SIZE_FOR_LR_DECAY,
    DEFAULT_PATIENCE_FOR_LR_DECAY,
    DEFAULT_MULTIPLIER_FOR_LR_DECAY,
    DEFAULT_REVERSE_ANNEALING_PERIOD
) {}

AdvantageActorCritic::AdvantageActorCritic(
    std::shared_ptr<DecisionNetHandler> handler
) : AdvantageActorCritic(handler, DEFAULT_LEARNING_RATE) {}

torch::Tensor AdvantageActorCritic::get_loss_from_logProba(
    const std::vector<
        std::unordered_map<Agent*, torch::Tensor>
    >& logProbas,
    Adam& optimizer,
    Agent* agent,
    const torch::Tensor& advantage
) {
    auto loss = torch::tensor(0.0, torch::requires_grad(true));
    for (int t = 0; t < advantage.size(0); t++) {
        auto logProbaSearch = logProbas[t].find(agent);
        if (logProbaSearch != logProbas[t].end()) {
            auto logProba = logProbaSearch->second;
            if (!std::isnan(logProba.item<double>())) {
                // nan values mean that we shouldn't train on this datum
                loss = loss + logProba * advantage[t];
            }
        }
        else {
            util::print_status(
                agent, "WARNING: Can't find in log probas at time " + std::to_string(t)
            );
        }
    }
    return loss;
}

std::pair<torch::Tensor, torch::Tensor> AdvantageActorCritic::get_advantage_for_person(
    Person* person
) {
    auto person_as_utilmaxer = static_cast<UtilMaxer*>(person);

    // Calculate advantage time series
    // advantage is realized utility minus predicted value in each state
    // Then critic loss is sum of squared advantage
    auto loss = torch::tensor(0.0);
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
            loss = loss + advantage_t.pow(2);
        }
        else {
            util::print_status(
                person,
                "WARNING: Can't find in rewards || value map at time " + std::to_string(t)
            );
        }
    }
    return std::make_pair(loss, advantage);
}

double AdvantageActorCritic::get_loss_for_person_in_episode(
    std::weak_ptr<Person> person,
    unsigned int numTotalPersons
) {
    auto person_shared = person.lock();
    assert(person_shared != nullptr);
    Person* person_ = person_shared.get();

    auto loss_advantage_pair = get_advantage_for_person(person_);
    auto loss = loss_advantage_pair.first;
    auto advantage = loss_advantage_pair.second;

    // Get loss from the other person decision nets
    torch::Tensor purchaseLoss = get_loss_from_logProba(
        handler->purchaseNetLogProba,
        purchaseNetOptim,
        person_,
        advantage
    );

    torch::Tensor laborSearchLoss = get_loss_from_logProba(
        handler->laborSearchNetLogProba,
        laborSearchNetOptim,
        person_,
        advantage
    );

    torch::Tensor consumptionLoss = get_loss_from_logProba(
        handler->consumptionNetLogProba,
        consumptionNetOptim,
        person_,
        advantage
    );

    loss = loss
            + purchaseLoss
            + laborSearchLoss
            + consumptionLoss;
    
    // also record intermediate loss values
    {
        std::lock_guard<std::mutex> lock(myMutex);
        purchaseNetLoss += purchaseLoss.item<double>();
        laborSearchNetLoss += laborSearchLoss.item<double>();
        consumptionNetLoss += consumptionLoss.item<double>();
    }

    loss.backward({}, true);
    return loss.item<double>();
}

std::pair<torch::Tensor, torch::Tensor> AdvantageActorCritic::get_advantage_for_firm(
    Firm* firm
) {
    // here advantage is realized profit minus predicted value in each state
    auto loss = torch::tensor(0.0, torch::requires_grad(true));
    auto advantage = torch::empty(handler->time - 1);
    auto q = torch::tensor(0.0, torch::requires_grad(true));
    // Note: we can't use firm's data from last period,
    // since we don't get to see the payoff for its decisions in that period
    for (int t = handler->time - 2; t >= 0; t--) {
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
            loss = loss + advantage_t.pow(2);
        }
        else {
            util::print_status(
                firm,
                "WARNING: Can't find in rewards || value map at time " + std::to_string(t)
            );
        }
    }
    return std::make_pair(loss, advantage);
}

double AdvantageActorCritic::get_loss_for_firm_in_episode(
    std::weak_ptr<Firm> firm,
    unsigned int numTotalFirms
) {
    auto firm_shared = firm.lock();
    assert(firm_shared != nullptr);
    Firm* firm_ = firm_shared.get();

    auto loss_advantage_pair = get_advantage_for_firm(firm_);
    auto loss = loss_advantage_pair.first;
    auto advantage = loss_advantage_pair.second;

    firmValueNetLoss += loss.item<double>();

    // Get loss from the other firm decision nets    
    torch::Tensor firmPurchaseLoss = get_loss_from_logProba(
        handler->firmPurchaseNetLogProba,
        firmPurchaseNetOptim,
        firm_,
        advantage
    );
    
    torch::Tensor productionLoss = get_loss_from_logProba(
        handler->productionNetLogProba,
        productionNetOptim,
        firm_,
        advantage
    );
    
    torch::Tensor offerLoss = get_loss_from_logProba(
        handler->offerNetLogProba,
        offerNetOptim,
        firm_,
        advantage
    );
    
    torch::Tensor jobOfferLoss = get_loss_from_logProba(
        handler->jobOfferNetLogProba,
        jobOfferNetOptim,
        firm_,
        advantage
    );

    loss = loss
            + firmPurchaseLoss
            + productionLoss
            + offerLoss
            + jobOfferLoss;
    
    // also record intermediate loss values
    {
        std::lock_guard<std::mutex> lock(myMutex);
        firmPurchaseNetLoss += firmPurchaseLoss.item<double>();
        productionNetLoss += productionLoss.item<double>();
        offerNetLoss += offerLoss.item<double>();
        jobOfferNetLoss += jobOfferLoss.item<double>();
    }

    loss.backward({}, true);
    return loss.item<double>();

}

void AdvantageActorCritic::get_loss_for_persons_multithreaded_(
    const std::vector<std::weak_ptr<Person>>& persons,
    unsigned int startIdx,
    unsigned int endIdx,
    double* loss
) {
    double loss_to_add = 0.0;
    for (unsigned int i = startIdx; i < endIdx; i++) {
        loss_to_add += get_loss_for_person_in_episode(persons[i], persons.size());
    }
    {
        std::lock_guard<std::mutex> lock(myMutex);
        *loss += loss_to_add;
    }
}

void AdvantageActorCritic::get_loss_for_firms_multithreaded_(
    const std::vector<std::weak_ptr<Firm>>& firms,
    unsigned int startIdx,
    unsigned int endIdx,
    double* loss
) {
    double loss_to_add = 0.0;
    for (unsigned int i = startIdx; i < endIdx; i++) {
        loss_to_add += get_loss_for_firm_in_episode(firms[i], firms.size());
    }
    {
        std::lock_guard<std::mutex> lock(myMutex);
        *loss += loss_to_add;
    }
}

double AdvantageActorCritic::get_loss_for_persons_multithreaded() {
    double loss = 0.0;
    auto persons = handler->economy->get_persons();
    std::vector<unsigned int> indices = util::get_indices_for_multithreading(persons.size());
    std::vector<std::thread> threads;
    threads.reserve(constants::numThreads);
    for (unsigned int i = 0; i < constants::numThreads; i++) {
        if (indices[i] != indices[i+1]) {
            threads.push_back(
                std::thread(
                    &AdvantageActorCritic::get_loss_for_persons_multithreaded_,
                    this,
                    persons,
                    indices[i],
                    indices[i+1],
                    &loss
                )
            );
        }
    }
    for (unsigned int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }
    return loss;
};

double AdvantageActorCritic::get_loss_for_firms_multithreaded() {
    double loss = 0.0;
    auto firms = handler->economy->get_firms();
    std::vector<unsigned int> indices = util::get_indices_for_multithreading(firms.size());
    std::vector<std::thread> threads;
    threads.reserve(constants::numThreads);
    for (unsigned int i = 0; i < constants::numThreads; i++) {
        if (indices[i] != indices[i+1]) {
            threads.push_back(
                std::thread(
                    &AdvantageActorCritic::get_loss_for_firms_multithreaded_,
                    this,
                    firms,
                    indices[i],
                    indices[i+1],
                    &loss
                )
            );
        }
    }
    for (unsigned int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }
    return loss;
}

void AdvantageActorCritic::zero_all_grads() {
    purchaseNetOptim.zero_grad();
    firmPurchaseNetOptim.zero_grad();
    laborSearchNetOptim.zero_grad();
    consumptionNetOptim.zero_grad();
    productionNetOptim.zero_grad();
    offerNetOptim.zero_grad();
    jobOfferNetOptim.zero_grad();
    valueNetOptim.zero_grad();
    firmValueNetOptim.zero_grad();
}

void AdvantageActorCritic::zero_all_tracked_losses() {
    purchaseNetLoss = 0.0;
    firmPurchaseNetLoss = 0.0;
    laborSearchNetLoss = 0.0;
    consumptionNetLoss = 0.0;
    productionNetLoss = 0.0;
    offerNetLoss = 0.0;
    jobOfferNetLoss = 0.0;
    valueNetLoss = 0.0;
    firmValueNetLoss = 0.0;
}

void AdvantageActorCritic::update_lr_schedulers() {
    purchaseNetScheduler.update_lr(purchaseNetLoss);
    firmPurchaseNetScheduler.update_lr(firmPurchaseNetLoss);
    laborSearchNetScheduler.update_lr(laborSearchNetLoss);
    consumptionNetScheduler.update_lr(consumptionNetLoss);
    productionNetScheduler.update_lr(productionNetLoss);
    offerNetScheduler.update_lr(offerNetLoss);
    jobOfferNetScheduler.update_lr(jobOfferNetLoss);
    valueNetScheduler.update_lr(valueNetLoss);
    firmValueNetScheduler.update_lr(firmValueNetLoss);
    zero_all_tracked_losses();
}

void AdvantageActorCritic::all_optims_step() {
    purchaseNetOptim.step();
    firmPurchaseNetOptim.step();
    laborSearchNetOptim.step();
    consumptionNetOptim.step();
    productionNetOptim.step();
    offerNetOptim.step();
    jobOfferNetOptim.step();
    valueNetOptim.step();
    firmValueNetOptim.step();
}

double AdvantageActorCritic::train_on_episode() {
    zero_all_grads();
    update_lr_schedulers();
    double loss_ = 0.0;
    if (constants::multithreaded) {
        loss_ += get_loss_for_persons_multithreaded();
        loss_ += get_loss_for_firms_multithreaded();
    }
    else {
        auto persons = handler->economy->get_persons();
        for (auto person : persons) {
            loss_ += get_loss_for_person_in_episode(person, persons.size());
        }
        auto firms = handler->economy->get_firms();
        for (auto firm : firms) {
            loss_ += get_loss_for_firm_in_episode(firm, firms.size());
        }
    }
    
    all_optims_step();
    return loss_;
}


} // namespace neural
