#include "advantageActorCritic.h"


namespace neural {

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

AdvantageActorCritic::AdvantageActorCritic(
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
            loss = loss + logProba * advantage[t];
        }
        else {
            print_status(
                agent, "WARNING: Can't find in log probas at time " + std::to_string(t)
            );
        }
    }
    return loss / handler->time;
}

torch::Tensor AdvantageActorCritic::get_loss_for_person_in_episode(
    std::shared_ptr<Person> person
) {
    auto person_as_utilmaxer = std::static_pointer_cast<UtilMaxer>(person);

    // Calculate advantage time series
    // advantage is realized utility minus predicted value in each state
    // Then critic loss is sum of squared advantage
    auto loss = torch::tensor(0.0, torch::requires_grad(true));
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

    // normalize by episode length
    loss = loss / handler->time;
    // std::cout << "After person adv: " << loss.item<float>() << std::endl;

    // Get loss from the other person decision nets
    loss = loss + get_loss_from_logProba(
        handler->purchaseNetLogProba,
        purchaseNetOptim,
        person,
        advantage
    );
    // std::cout << "After person purchaseNet: " << loss.item<float>() << std::endl;
    loss = loss + get_loss_from_logProba(
        handler->laborSearchNetLogProba,
        laborSearchNetOptim,
        person,
        advantage
    );
    // std::cout << "After person laborSearchNet: " << loss.item<float>() << std::endl;
    loss = loss + get_loss_from_logProba(
        handler->consumptionNetLogProba,
        consumptionNetOptim,
        person,
        advantage
    );
    // std::cout << "After person consumptionNet: " << loss.item<float>() << std::endl;

    return loss;
}

torch::Tensor AdvantageActorCritic::get_loss_for_firm_in_episode(
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
                std::static_pointer_cast<ProfitMaxer>(firm),
                "WARNING: Can't find in rewards || value map at time " + std::to_string(t)
            );
        }
    }

    loss = loss / (handler->time - 1);
    // std::cout << "After firm adv: " << loss.item<float>() << std::endl;

    // Get loss from the other firm decision nets
    loss = loss + get_loss_from_logProba(
        handler->firmPurchaseNetLogProba,
        firmPurchaseNetOptim,
        firm,
        advantage
    );
    // std::cout << "After firm purchaseNet: " << loss.item<float>() << std::endl;
    loss = loss + get_loss_from_logProba(
        handler->productionNetLogProba,
        productionNetOptim,
        firm,
        advantage
    );
    // std::cout << "After firm productionNet: " << loss.item<float>() << std::endl;
    loss = loss + get_loss_from_logProba(
        handler->offerNetLogProba,
        offerNetOptim,
        firm,
        advantage
    );
    // std::cout << "After firm offerNet: " << loss.item<float>() << std::endl;
    loss = loss + get_loss_from_logProba(
        handler->jobOfferNetLogProba,
        jobOfferNetOptim,
        firm,
        advantage
    );
    // std::cout << "After firm jobOfferNet: " << loss.item<float>() << std::endl;

    return loss;
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
    auto loss = torch::tensor(0.0);
    for (auto person : handler->economy->get_persons()) {
        loss = loss + get_loss_for_person_in_episode(person);
    }
    for (auto firm : handler->economy->get_firms()) {
        loss = loss + get_loss_for_firm_in_episode(firm);
    }
    // normalize loss by number of agents in the economy
    loss = loss / static_cast<long>(handler->economy->get_maxAgents());
    loss.backward();
    all_optims_step();
    return loss.item<float>();
}


} // namespace neural
