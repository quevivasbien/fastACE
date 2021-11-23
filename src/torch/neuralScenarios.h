#ifndef NEURAL_SCENARIOS_H
#define NEURAL_SCENARIOS_H

#include <memory>
#include <chrono>
#include <math.h>
#include "scenario.h"
#include "neuralEconomy.h"
#include "neuralPersonDecisionMaker.h"
#include "neuralFirmDecisionMaker.h"
#include "utilMaxer.h"
#include "profitMaxer.h"
#include "advantageActorCritic.h"
#include "util.h"
#include "constants.h"


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


struct VariablePopulationScenario : NeuralScenario {
    VariablePopulationScenario(
        unsigned int numPeople,
        unsigned int numFirms
    ) : NeuralScenario(), numPeople(numPeople), numFirms(numFirms) {}
    VariablePopulationScenario(
        std::shared_ptr<AdvantageActorCritic> trainer,
        unsigned int numPeople,
        unsigned int numFirms
    ) : NeuralScenario(trainer), numPeople(numPeople), numFirms(numFirms) {}

    virtual std::shared_ptr<Economy> setup() {
        std::shared_ptr<NeuralEconomy> economy = get_economy({"bread", "capital"}, numPeople + numFirms);

        auto rng = economy->get_rng();
        std::normal_distribution<double> randn(0, 1);

        for (unsigned int i = 0; i < numPeople; i++) {
            // need to generate a value in [0, 1] near 1 for discount rate
            double discountParam = 3 + randn(rng);
            double discountRate = 1 / (1 + discountParam * discountParam);
            // elasticity of subtitution must be > 0
            double elasticity = 5.0 + randn(rng);
            if (elasticity <= 0) {
                elasticity = constants::eps;
            }
            UtilMaxer::init(
                economy,
                // Everyone starts with same goods and money
                Eigen::Array2d(10.0, 10.0),
                20.0,
                std::make_shared<CES>(
                    1.0,
                    Eigen::Array3d(0.5 + randn(rng)*0.1, 0.1 + randn(rng)*0.05, 0.75 + randn(rng)*0.1),
                    10.0 + randn(rng)
                ),
                discountRate,
                std::make_shared<NeuralPersonDecisionMaker>(economy->handler)
            );
        }

        for (unsigned int i = 0; i < numFirms; i++) {
            // elasticity of subtitution must be > 0
            double elasticity1 = 5.0 + randn(rng);
            double elasticity2 = 5.0 + randn(rng);
            ProfitMaxer::init(
                economy,
                std::vector<std::shared_ptr<Agent>>({nullptr}),
                Eigen::Array2d(10.0, 20.0),
                50.0,
                create_CES_VecToVec(
                    std::vector<double>({0.5, 1.0}),
                    std::vector<Eigen::ArrayXd>({
                        Eigen::Array3d(1.0 + randn(rng)*0.1, 0.0, 1.0 + randn(rng)*0.1),
                        Eigen::Array3d(1.0 + randn(rng)*0.1, 0.0, 1.0 + randn(rng)*0.1)
                    }),
                    std::vector<double>({elasticity1, elasticity2})
                ),
                std::make_shared<NeuralFirmDecisionMaker>(economy->handler)
            );
        }

        return economy;
    }

    unsigned int numPeople;
    unsigned int numFirms;
};


inline std::vector<float> train(
    std::shared_ptr<NeuralScenario> scenario,
    unsigned int numEpisodes,
    unsigned int episodeLength,
    unsigned int updateEveryNEpisodes
) {
    auto start = std::chrono::system_clock::now();
    std::shared_ptr<Economy> economy;
    std::vector<float> losses(numEpisodes);
    for (unsigned int i = 0; i < numEpisodes; i++) {
        economy = scenario->setup();

        auto step_time_start = std::chrono::system_clock::now();
        for (unsigned int t = 0; t < episodeLength; t++) {
            economy->time_step();
        }
        auto step_time_end = std::chrono::system_clock::now();

        float loss = scenario->trainer->train_on_episode();
        auto train_time_end = std::chrono::system_clock::now();

        pprint(2, "Time spent time stepping:");
        pprint_time_elasped(2, step_time_start, step_time_end);
        pprint(2, "Time spent time training:");
        pprint_time_elasped(2, step_time_end, train_time_end);

        // TODO: Come up with a better checkpointing system
        if (!isnan(loss)) {
            scenario->handler->save_models();
        }
        else if (i > 0) {
            pprint(1, "NaN encountered; loading from checkpoint.");
            scenario->handler->load_models();
            loss = losses[i-1];
        }
        else {
            // need to start over
            std::cout << "Training failed on first episode.\n";
            break;
        }
        losses[i] = loss;

        if ((updateEveryNEpisodes != 0) && ((i + 1) % updateEveryNEpisodes == 0)) {
            float sum = 0.0;
            for (int j = 0; j < updateEveryNEpisodes; j++) {
                sum += losses[i - j];
            }
            print(
                "Episode " + std::to_string(i+1)
                + ": Average loss over past " + std::to_string(updateEveryNEpisodes)
                + " episodes = " + std::to_string(sum / updateEveryNEpisodes)
            );
        }
    }
    auto end = std::chrono::system_clock::now();
    pprint(1, "Total time:");
    pprint_time_elasped(1, start, end);

    return losses;
}

inline std::vector<float> train(
    std::shared_ptr<NeuralScenario> scenario,
    unsigned int numEpisodes,
    unsigned int episodeLength
) {
    unsigned int updateEveryNEpisodes = numEpisodes / 10;
    if (updateEveryNEpisodes == 0) {
        updateEveryNEpisodes++;
    }
    return train(scenario, numEpisodes, episodeLength, updateEveryNEpisodes);
}

} // namespace neural



#endif