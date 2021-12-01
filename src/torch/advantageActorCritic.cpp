#include "advantageActorCritic.h"


namespace neural {



LRScheduler::LRScheduler(
    std::shared_ptr<torch::optim::Adam> optimizer,
    unsigned int episodeBatchSize,
    unsigned int patience,
    float decayMultiplier,
    std::string name
) : optimizer(optimizer),
    episodeBatchSize(episodeBatchSize),
    patience(patience),
    decayMultiplier(decayMultiplier),
    name(name)
{}

void LRScheduler::decay_lr() {
    pprint(1, "Decaying LR for " + name);
    for (auto& group : optimizer->param_groups()) {
        if (group.has_options()) {
            auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
            options.lr(options.lr() * decayMultiplier);
        }
    }
}

void LRScheduler::update_lr(float loss) {
    lossHistory.push_back(loss);

    if (lossHistory.size() == episodeBatchSize) {
        float recentBatchLoss = 0.0;
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
            decay_lr();
            numBadBatches = 0;
        }

        lossHistory.clear();
    }
}


AdvantageActorCritic::AdvantageActorCritic(
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
    ),

    purchaseNetScheduler(LRScheduler(
        purchaseNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        "purchaseNet"
    )),
    firmPurchaseNetScheduler(LRScheduler(
        firmPurchaseNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        "firmPurchaseNet"
    )),
    laborSearchNetScheduler(LRScheduler(
        laborSearchNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        "laborSearchNet"
    )),
    consumptionNetScheduler(LRScheduler(
        consumptionNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        "consumptionNet"
    )),
    productionNetScheduler(LRScheduler(
        productionNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        "productionNet"
    )),
    offerNetScheduler(LRScheduler(
        offerNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        "offerNet"
    )),
    jobOfferNetScheduler(LRScheduler(
        jobOfferNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        "jobOfferNet"
    )),
    valueNetScheduler(LRScheduler(
        valueNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        "valueNet"
    )),
    firmValueNetScheduler(LRScheduler(
        firmValueNetOptim,
        episodeBatchSizeForLRDecay,
        patienceForLRDecay,
        multiplierForLRDecay,
        "firmValueNet"
    ))
{}

AdvantageActorCritic::AdvantageActorCritic(
    std::shared_ptr<DecisionNetHandler> handler,
    float initialLR
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
    DEFAULT_MULTIPLIER_FOR_LR_DECAY
) {}

AdvantageActorCritic::AdvantageActorCritic(
    std::shared_ptr<DecisionNetHandler> handler
) : AdvantageActorCritic(handler, DEFAULT_LEARNING_RATE) {}

torch::Tensor AdvantageActorCritic::get_loss_from_logProba(
    const std::vector<
        std::unordered_map<std::shared_ptr<Agent>, torch::Tensor>
    >& logProbas,
    std::shared_ptr<Adam> optimizer,
    std::shared_ptr<Agent> agent,
    const torch::Tensor& advantage
) {
    auto loss = torch::tensor(0.0, torch::requires_grad(true));
    for (int t = 0; t < advantage.size(0); t++) {
        auto logProbaSearch = logProbas[t].find(agent);
        if (logProbaSearch != logProbas[t].end()) {
            auto logProba = logProbaSearch->second;
            // std::cout << "logProba: " << logProba.item<float>() << std::endl;
            if (!std::isnan(logProba.item<float>())) {
                // nan values mean that we shouldn't train on this datum
                loss = loss + logProba * advantage[t];
            }
        }
        else {
            print_status(
                agent, "WARNING: Can't find in log probas at time " + std::to_string(t)
            );
        }
    }
    return loss;
}

std::pair<torch::Tensor, torch::Tensor> AdvantageActorCritic::get_advantage_for_person(
    std::shared_ptr<Person> person
) {
    auto person_as_utilmaxer = std::static_pointer_cast<UtilMaxer>(person);

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
            print_status(
                person_as_utilmaxer,
                "WARNING: Can't find in rewards || value map at time " + std::to_string(t)
            );
        }
    }
    return std::make_pair(loss, advantage);
}

float AdvantageActorCritic::get_loss_for_person_in_episode(
    std::shared_ptr<Person> person,
    unsigned int numTotalPersons
) {
    auto loss_advantage_pair = get_advantage_for_person(person);
    auto loss = loss_advantage_pair.first;
    auto advantage = loss_advantage_pair.second;

    // Get loss from the other person decision nets
    torch::Tensor purchaseLoss = get_loss_from_logProba(
        handler->purchaseNetLogProba,
        purchaseNetOptim,
        person,
        advantage
    );

    torch::Tensor laborSearchLoss = get_loss_from_logProba(
        handler->laborSearchNetLogProba,
        laborSearchNetOptim,
        person,
        advantage
    );

    torch::Tensor consumptionLoss = get_loss_from_logProba(
        handler->consumptionNetLogProba,
        consumptionNetOptim,
        person,
        advantage
    );

    loss = loss
            + purchaseLoss
            + laborSearchLoss
            + consumptionLoss;
    
    // also record intermediate loss values
    {
        std::lock_guard<std::mutex> lock(myMutex);
        purchaseNetLoss += purchaseLoss.item<float>();
        laborSearchNetLoss += laborSearchLoss.item<float>();
        consumptionNetLoss += consumptionLoss.item<float>();
    }

    // normalize by episode length and num persons in economy
    loss = loss / static_cast<long>(handler->time * numTotalPersons);
    loss.backward({}, true);
    return loss.item<float>();
}

std::pair<torch::Tensor, torch::Tensor> AdvantageActorCritic::get_advantage_for_firm(
    std::shared_ptr<Firm> firm
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
            print_status(
                firm,
                "WARNING: Can't find in rewards || value map at time " + std::to_string(t)
            );
        }
    }
    return std::make_pair(loss, advantage);
}

float AdvantageActorCritic::get_loss_for_firm_in_episode(
    std::shared_ptr<Firm> firm,
    unsigned int numTotalFirms
) {
    auto loss_advantage_pair = get_advantage_for_firm(firm);
    auto loss = loss_advantage_pair.first;
    auto advantage = loss_advantage_pair.second;

    firmValueNetLoss += loss.item<float>();

    // Get loss from the other firm decision nets    
    torch::Tensor firmPurchaseLoss = get_loss_from_logProba(
        handler->firmPurchaseNetLogProba,
        firmPurchaseNetOptim,
        firm,
        advantage
    );
    
    torch::Tensor productionLoss = get_loss_from_logProba(
        handler->productionNetLogProba,
        productionNetOptim,
        firm,
        advantage
    );
    
    torch::Tensor offerLoss = get_loss_from_logProba(
        handler->offerNetLogProba,
        offerNetOptim,
        firm,
        advantage
    );
    
    torch::Tensor jobOfferLoss = get_loss_from_logProba(
        handler->jobOfferNetLogProba,
        jobOfferNetOptim,
        firm,
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
        firmPurchaseNetLoss += firmPurchaseLoss.item<float>();
        productionNetLoss += productionLoss.item<float>();
        offerNetLoss += offerLoss.item<float>();
        jobOfferNetLoss += jobOfferLoss.item<float>();
    }

    loss = loss / static_cast<long>((handler->time - 1) * numTotalFirms);
    loss.backward({}, true);
    return loss.item<float>();

}

void AdvantageActorCritic::get_loss_for_persons_multithreaded_(
    const std::vector<std::shared_ptr<Person>>& persons,
    unsigned int startIdx,
    unsigned int endIdx,
    float* loss
) {
    float loss_to_add = 0.0;
    for (unsigned int i = startIdx; i < endIdx; i++) {
        loss_to_add += get_loss_for_person_in_episode(persons[i], persons.size());
    }
    {
        std::lock_guard<std::mutex> lock(myMutex);
        *loss += loss_to_add;
    }
}

void AdvantageActorCritic::get_loss_for_firms_multithreaded_(
    const std::vector<std::shared_ptr<Firm>>& firms,
    unsigned int startIdx,
    unsigned int endIdx,
    float* loss
) {
    float loss_to_add = 0.0;
    for (unsigned int i = startIdx; i < endIdx; i++) {
        loss_to_add += get_loss_for_firm_in_episode(firms[i], firms.size());
    }
    {
        std::lock_guard<std::mutex> lock(myMutex);
        *loss += loss_to_add;
    }
}

float AdvantageActorCritic::get_loss_for_persons_multithreaded() {
    float loss = 0.0;
    auto persons = handler->economy->get_persons();
    std::vector<unsigned int> indices = get_indices_for_multithreading(persons.size());
    std::vector<std::thread> threads;
    threads.reserve(constants::config.numThreads);
    for (unsigned int i = 0; i < constants::config.numThreads; i++) {
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

float AdvantageActorCritic::get_loss_for_firms_multithreaded() {
    float loss = 0.0;
    auto firms = handler->economy->get_firms();
    std::vector<unsigned int> indices = get_indices_for_multithreading(firms.size());
    std::vector<std::thread> threads;
    threads.reserve(constants::config.numThreads);
    for (unsigned int i = 0; i < constants::config.numThreads; i++) {
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

float AdvantageActorCritic::get_loss_multithreaded() {
    auto persons = handler->economy->get_persons();
    unsigned int numPersons = persons.size();
    auto firms = handler->economy->get_firms();
    unsigned int numFirms = firms.size();

    auto loss = torch::tensor(0.0);
    // first get advantage and loss for each person
    std::vector<torch::Tensor> personAdvantages(numPersons);
    for (unsigned int i = 0; i < numPersons; i++) {
        auto loss_advantage_pair = get_advantage_for_person(persons[i]);
        loss = loss + loss_advantage_pair.first;
        personAdvantages[i] = loss_advantage_pair.second;
    }
    // same idea for firms
    std::vector<torch::Tensor> firmAdvantages(numFirms);
    for (unsigned int i = 0; i < numFirms; i++) {
        auto loss_advantage_pair = get_advantage_for_firm(firms[i]);
        loss = loss_advantage_pair.first;
        firmAdvantages[i] = loss_advantage_pair.second;
    }
    loss.backward({}, true);
    float loss_ = loss.item<float>();

    // now go through the decision nets, multithreaded
    // person decision nets
    std::thread purchaseThread(
        &AdvantageActorCritic::get_loss_from_logProba_multithreaded<Person>,
        this,
        numPersons,
        handler->purchaseNetLogProba,
        purchaseNetOptim,
        persons,
        personAdvantages,
        &loss_
    );
    std::thread laborSearchThread(
        &AdvantageActorCritic::get_loss_from_logProba_multithreaded<Person>,
        this,
        numPersons,
        handler->laborSearchNetLogProba,
        laborSearchNetOptim,
        persons,
        personAdvantages,
        &loss_
    );
    std::thread consumptionThread(
        &AdvantageActorCritic::get_loss_from_logProba_multithreaded<Person>,
        this,
        numPersons,
        handler->consumptionNetLogProba,
        consumptionNetOptim,
        persons,
        personAdvantages,
        &loss_
    );
    // firm decision nets
    std::thread firmPurchaseThread(
        &AdvantageActorCritic::get_loss_from_logProba_multithreaded<Firm>,
        this,
        numFirms,
        handler->firmPurchaseNetLogProba,
        firmPurchaseNetOptim,
        firms,
        firmAdvantages,
        &loss_
    );
    std::thread productionThread(
        &AdvantageActorCritic::get_loss_from_logProba_multithreaded<Firm>,
        this,
        numFirms,
        handler->productionNetLogProba,
        productionNetOptim,
        firms,
        firmAdvantages,
        &loss_
    );
    std::thread offerThread(
        &AdvantageActorCritic::get_loss_from_logProba_multithreaded<Firm>,
        this,
        numFirms,
        handler->offerNetLogProba,
        offerNetOptim,
        firms,
        firmAdvantages,
        &loss_
    );
    std::thread jobOfferThread(
        &AdvantageActorCritic::get_loss_from_logProba_multithreaded<Firm>,
        this,
        numFirms,
        handler->jobOfferNetLogProba,
        jobOfferNetOptim,
        firms,
        firmAdvantages,
        &loss_
    );

    // join all threads and finish
    purchaseThread.join();
    firmPurchaseThread.join();
    laborSearchThread.join();
    consumptionThread.join();
    productionThread.join();
    offerThread.join();
    jobOfferThread.join();

    return loss_;
}

void AdvantageActorCritic::zero_all_grads() {
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

float AdvantageActorCritic::train_on_episode() {
    zero_all_grads();
    update_lr_schedulers();
    float loss_ = 0.0;
    if (constants::config.multithreaded) {
        // loss_ += get_loss_multithreaded();
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
