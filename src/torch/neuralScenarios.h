#ifndef NEURAL_SCENARIOS_H
#define NEURAL_SCENARIOS_H

#include <memory>
#include <chrono>
#include <cmath>
#include "scenario.h"
#include "neuralEconomy.h"
#include "utilMaxer.h"
#include "profitMaxer.h"
#include "advantageActorCritic.h"
#include "util.h"
#include "constants.h"
#include "neuralConstants.h"


namespace neural {

// a NeuralScenario returns a neuralEconomy rather than a base Economy
struct NeuralScenario : Scenario {
    NeuralScenario() : handler(nullptr), trainer(nullptr) {}
    NeuralScenario(
        std::shared_ptr<AdvantageActorCritic> trainer
    ) : handler(trainer->handler), trainer(trainer) {}

    std::shared_ptr<DecisionNetHandler> handler;
    std::shared_ptr<AdvantageActorCritic> trainer;

    std::shared_ptr<NeuralEconomy> get_economy(
        std::vector<std::string> goods,
        unsigned int maxAgents
    ) {
        std::shared_ptr<NeuralEconomy> economy;
        // if handler not yet initialized, initialize as default handler
        if (handler == nullptr) {
            economy = NeuralEconomy::init(goods, maxAgents);
            handler = economy->handler;
        }
        else {
            economy = NeuralEconomy::init(goods, maxAgents, handler);
            handler->reset(economy);
        }
        // also make sure trainer is initialized
        if (trainer == nullptr) {
            trainer = std::make_shared<AdvantageActorCritic>(handler);
        }
        return economy;
    }
};


struct SimpleScenario : NeuralScenario {
    // A basic scenario with two persons and one firm, helpful for testing/debugging
    SimpleScenario() : NeuralScenario() {}
    SimpleScenario(
        std::shared_ptr<AdvantageActorCritic> trainer
    ) : NeuralScenario(trainer) {}

    virtual std::shared_ptr<Economy> setup() {
        std::shared_ptr<NeuralEconomy> economy = get_economy({"bread", "capital"}, 3);

        UtilMaxer::init(
            economy,
            Eigen::Array2d(10.0, 10.0),
            20.0,
            std::make_shared<CES>(1.0, Eigen::Array3d(0.5, 0.5, 0.5), 1.3),
            0.8,
            std::make_shared<NeuralPersonDecisionMaker>(economy->handler)
        );
        UtilMaxer::init(
            economy,
            Eigen::Array2d(10.0, 10.0),
            20.0,
            std::make_shared<CES>(1.0, Eigen::Array3d(0.2, 0.6, 0.4), 1.3),
            0.9,
            std::make_shared<NeuralPersonDecisionMaker>(economy->handler)
        );

        ProfitMaxer::init(
            economy,
            std::vector<std::shared_ptr<Agent>>({nullptr}),
            Eigen::Array2d(10.0, 20.0),
            50.0,
            create_CES_VecToVec(
                std::vector<double>({0.5, 1.0}),
                std::vector<Eigen::ArrayXd>({
                    Eigen::Array3d(1.0, 0.0, 1.0),
                    Eigen::Array3d(1.0, 0.0, 1.0)
                }),
                std::vector<double>({3.0, 5.0})
            ),
            std::make_shared<NeuralFirmDecisionMaker>(economy->handler)
        );

        return economy;
    }
};


struct CustomScenarioParams {

    CustomScenarioParams(
        unsigned int numPeople,
        unsigned int numFirms
    ) : numPeople(numPeople), numFirms(numFirms) {}

    unsigned int numPeople;
    unsigned int numFirms;

    // params for people

    double money_mu = 10.0;
    double money_sigma = 2.0;
    double good1_mu = 10.0;
    double good1_sigma = 2.0;
    double good2_mu = 1.0;
    double good2_sigma = 0.2;

    // it's okay if these don't sum to 1 as they will be normalized
    double labor_share_mu = 0.4;
    double labor_share_sigma = 0.1;
    double good1_share_mu = 0.4;
    double good1_share_sigma = 0.1;
    double good2_share_mu = 0.1;
    double good2_share_sigma = 0.02;

    // will be plugged into logistic function
    double discount_mu = 2.0;
    double discount_sigma = 1.0;

    double elasticity_mu = 10.0;
    double elasticity_sigma = 2.5;

    // params for firms

    double firm_money_mu = 50.0;
    double firm_money_sigma = 10.0;
    double firm_good1_mu = 10.0;
    double firm_good1_sigma = 4.0;
    double firm_good2_mu = 30.0;
    double firm_good2_sigma = 5.0;

    double firm_tfp1_mu = 1.0;
    double firm_tfp1_sigma = 0.2;
    double firm_tfp2_mu = 1.0;
    double firm_tfp2_sigma = 0.2;

    double firm_labor_share1_mu = 0.4;
    double firm_labor_share1_sigma = 0.05;
    double firm_good1_share1_mu = 0.1;
    double firm_good1_share1_sigma = 0.02;
    double firm_good2_share1_mu = 0.4;
    double firm_good2_share1_sigma = 0.02;

    double firm_labor_share2_mu = 0.4;
    double firm_labor_share2_sigma = 0.05;
    double firm_good1_share2_mu = 0.1;
    double firm_good1_share2_sigma = 0.02;
    double firm_good2_share2_mu = 0.4;
    double firm_good2_share2_sigma = 0.05;

    double firm_elasticity1_mu = 10.0;
    double firm_elasticity1_sigma = 2.5;
    double firm_elasticity2_mu = 10.0;
    double firm_elasticity2_sigma = 2.5;
};


struct CustomScenario : NeuralScenario {
    // A versatile scenario: create a struct of parameters just how you like it or use the default config ;)

    CustomScenario(
        unsigned int numPeople,
        unsigned int numFirms
    ) : NeuralScenario(), params(CustomScenarioParams(numPeople, numFirms)) {}
    
    CustomScenario(
        std::shared_ptr<AdvantageActorCritic> trainer,
        unsigned int numPeople,
        unsigned int numFirms
    ) : NeuralScenario(trainer), params(CustomScenarioParams(numPeople, numFirms)) {}
    
    CustomScenario(
        std::shared_ptr<AdvantageActorCritic> trainer,
        CustomScenarioParams params
    ) : NeuralScenario(trainer), params(params) {}

    virtual std::shared_ptr<Economy> setup() {
        std::shared_ptr<NeuralEconomy> economy = get_economy(
            {"bread", "capital"},
            params.numPeople + params.numFirms
        );

        auto rng = economy->get_rng();
        std::normal_distribution<double> randn(0, 1);

        for (unsigned int i = 0; i < params.numPeople; i++) {
            ;
            UtilMaxer::init(
                economy,
                Eigen::Array2d(
                    make_nonnegative(params.good1_mu + params.good1_sigma * randn(rng)),
                    make_nonnegative(params.good2_mu + params.good2_sigma * randn(rng))
                ),
                make_nonnegative(params.money_mu + params.money_sigma * randn(rng)),
                std::make_shared<CES>(
                    1.0,
                    Eigen::Array3d(
                        params.labor_share_mu + params.labor_share_sigma * randn(rng),
                        params.good1_share_mu + params.good1_share_sigma * randn(rng),
                        params.good2_share_mu + params.good2_share_sigma * randn(rng)
                    ),
                    make_positive(params.elasticity_mu + params.elasticity_sigma * randn(rng))
                ),
                // need to generate a value in [0, 1] for discount rate
                // use a "logit normal" distribution
                1.0 / (1.0 + exp(-params.discount_mu - params.discount_sigma * randn(rng))),
                std::make_shared<NeuralPersonDecisionMaker>(economy->handler)
            );
        }

        for (unsigned int i = 0; i < params.numFirms; i++) {
            ProfitMaxer::init(
                economy,
                std::vector<std::shared_ptr<Agent>>({nullptr}),
                Eigen::Array2d(
                    make_nonnegative(params.firm_good1_mu + params.firm_good2_sigma * randn(rng)),
                    make_nonnegative(params.firm_good2_mu + params.firm_good2_sigma * randn(rng))
                ),
                make_nonnegative(params.firm_money_mu + params.firm_money_sigma * randn(rng)),
                create_CES_VecToVec(
                    std::vector<double>({
                        make_nonnegative(params.firm_tfp1_mu + params.firm_tfp1_sigma * randn(rng)),
                        make_nonnegative(params.firm_tfp2_mu + params.firm_tfp2_sigma * randn(rng))
                    }),
                    std::vector<Eigen::ArrayXd>({
                        Eigen::Array3d(
                            params.firm_labor_share1_mu + params.firm_labor_share1_sigma * randn(rng),
                            params.firm_good1_share1_mu + params.firm_good1_share1_sigma * randn(rng),
                            params.firm_good2_share1_mu + params.firm_good2_share1_sigma * randn(rng)
                        ),
                        Eigen::Array3d(
                            params.firm_labor_share2_mu + params.firm_labor_share2_sigma * randn(rng),
                            params.firm_good1_share2_mu + params.firm_good1_share2_sigma * randn(rng),
                            params.firm_good2_share2_mu + params.firm_good2_share2_sigma * randn(rng)
                        )
                    }),
                    std::vector<double>({
                        make_positive(params.firm_elasticity1_mu + params.firm_elasticity1_sigma * randn(rng)),
                        make_positive(params.firm_elasticity2_mu + params.firm_elasticity2_sigma * randn(rng))
                    })
                ),
                std::make_shared<NeuralFirmDecisionMaker>(economy->handler)
            );
        }

        return economy;
    }

    // can be used to get a placeholder for the sort of economy generated by setup()
    // without actually having instantiated an instance of this struct
    static std::shared_ptr<NeuralEconomy> setup_dummy() {
        return NeuralEconomy::init_dummy(2);
    }

    CustomScenarioParams params;
};



struct TrainingParams {
    TrainingParams(
        unsigned int numEpisodes,
        unsigned int episodeLength,
        unsigned int updateEveryNEpisodes,
        unsigned int checkpointEveryNEpisodes
    ) : numEpisodes(numEpisodes),
        episodeLength(episodeLength),
        updateEveryNEpisodes(updateEveryNEpisodes),
        checkpointEveryNEpisodes(checkpointEveryNEpisodes)
    {}

    unsigned int numEpisodes;
    unsigned int episodeLength;
    unsigned int updateEveryNEpisodes;
    unsigned int checkpointEveryNEpisodes;

    unsigned int stackSize = DEFAULT_stackSize;
    unsigned int encodingSize = DEFAULT_encodingSize;
    unsigned int hiddenSize = DEFAULT_hiddenSize;
    unsigned int nHidden = DEFAULT_nHidden;
    unsigned int nHiddenSmall = DEFAULT_nHiddenSmall;

    float purchaseNetLR = DEFAULT_LEARNING_RATE;
    float firmPurchaseNetLR = DEFAULT_LEARNING_RATE;
    float laborSearchNetLR = DEFAULT_LEARNING_RATE;
    float consumptionNetLR = DEFAULT_LEARNING_RATE;
    float productionNetLR = DEFAULT_LEARNING_RATE;
    float offerNetLR = DEFAULT_LEARNING_RATE;
    float jobOfferNetLR = DEFAULT_LEARNING_RATE;
    float valueNetLR = DEFAULT_LEARNING_RATE;
    float firmValueNetLR = DEFAULT_LEARNING_RATE;
    unsigned int episodeBatchSizeForLRDecay = DEFAULT_EPISODE_BATCH_SIZE_FOR_LR_DECAY;
    unsigned int patienceForLRDecay = DEFAULT_PATIENCE_FOR_LR_DECAY;
    float multiplierForLRDecay = DEFAULT_MULTIPLIER_FOR_LR_DECAY;
};


inline std::vector<float> train(
    std::shared_ptr<NeuralScenario> scenario,
    const TrainingParams& params
) {
    // note: this implementation will ignore all TrainingParam members except the first four:
    // numEpisodes, episodeLength, updateEveryNEpisodes, checkPointEveryNEpisodes
    // other params are taken from scenario->trainer

    // If you want to specify more options, use the overloaded train function that takes as args
    // a CustomScenarioParams struct and a TrainingParams struct
    auto start = std::chrono::system_clock::now();
    std::shared_ptr<Economy> economy;
    std::vector<float> losses(params.numEpisodes);
    for (unsigned int i = 0; i < params.numEpisodes; i++) {
        economy = scenario->setup();

        auto step_time_start = std::chrono::system_clock::now();
        for (unsigned int t = 0; t < params.episodeLength; t++) {
            economy->time_step();
        }
        auto step_time_end = std::chrono::system_clock::now();

        float loss = scenario->trainer->train_on_episode();
        auto train_time_end = std::chrono::system_clock::now();

        pprint(2, "Time spent time stepping:");
        pprint_time_elasped(2, step_time_start, step_time_end);
        pprint(2, "Time spent time training:");
        pprint_time_elasped(2, step_time_end, train_time_end);

        if (std::isnan(loss)) {
            if (i >= params.checkpointEveryNEpisodes) {
                pprint(
                    1,
                    "In episode " + std::to_string(i+1) + ": NaN encountered; reverting to last checkpoint."
                );
                scenario->handler->load_models();
                loss = losses[i-1];
            }
            else {
                std::cout << "Training failed before first checkpoint.\n";
                break;
            }
        }
        else if (((i - 1) % params.checkpointEveryNEpisodes == 0) || (i == params.numEpisodes - 1)) {
            scenario->handler->save_models();
        }
        losses[i] = loss;

        if ((params.updateEveryNEpisodes != 0) && ((i + 1) % params.updateEveryNEpisodes == 0)) {
            float sum = 0.0;
            for (int j = 0; j < params.updateEveryNEpisodes; j++) {
                sum += losses[i - j];
            }
            print(
                "Episode " + std::to_string(i+1)
                + ": Average loss over past " + std::to_string(params.updateEveryNEpisodes)
                + " episodes = " + std::to_string(sum / params.updateEveryNEpisodes)
            );
        }
    }
    auto end = std::chrono::system_clock::now();
    pprint(1, "Total time:");
    pprint_time_elasped(1, start, end);

    return losses;
}

// this is just a convenience function if you don't want to create a TrainingParams object
inline std::vector<float> train(
    std::shared_ptr<NeuralScenario> scenario,
    unsigned int numEpisodes,
    unsigned int episodeLength,
    unsigned int updateEveryNEpisodes,
    unsigned int checkpointEveryNEpisodes
) {
    if (updateEveryNEpisodes == 0) {
        updateEveryNEpisodes++;
    }
    return train(
        scenario,
        TrainingParams(
            numEpisodes,
            episodeLength,
            updateEveryNEpisodes,
            checkpointEveryNEpisodes
        )
    );
}

// This is the easiest way to set up training in most cases
// Just supply a CustomScenarioParams struct and a TrainingParams struct
inline std::vector<float> train(
    const CustomScenarioParams& scenarioParams,
    const TrainingParams& trainingParams
) {
    auto handler = std::make_shared<DecisionNetHandler>(
        CustomScenario::setup_dummy(),
        trainingParams.stackSize,
        trainingParams.encodingSize,
        trainingParams.hiddenSize,
        trainingParams.nHidden,
        trainingParams.nHiddenSmall
    );
    auto trainer = std::make_shared<AdvantageActorCritic>(
        handler,
        trainingParams.purchaseNetLR,
        trainingParams.firmPurchaseNetLR,
        trainingParams.laborSearchNetLR,
        trainingParams.consumptionNetLR,
        trainingParams.productionNetLR,
        trainingParams.offerNetLR,
        trainingParams.jobOfferNetLR,
        trainingParams.valueNetLR,
        trainingParams.firmValueNetLR,
        trainingParams.episodeBatchSizeForLRDecay,
        trainingParams.patienceForLRDecay,
        trainingParams.multiplierForLRDecay
    );
    auto scenario = std::make_shared<CustomScenario>(trainer, scenarioParams);

    return train(scenario, trainingParams);
}

} // namespace neural



#endif