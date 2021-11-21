#ifndef NEURAL_SCENARIOS_H
#define NEURAL_SCENARIOS_H

#include <memory>
#include "scenario.h"
#include "neuralEconomy.h"
#include "neuralPersonDecisionMaker.h"
#include "neuralFirmDecisionMaker.h"
#include "utilMaxer.h"
#include "profitMaxer.h"
#include "advantageActorCritic.h"


namespace neural {

struct SimpleScenario : Scenario {
    SimpleScenario() : handler(nullptr), trainer(nullptr) {}
    SimpleScenario(
        std::shared_ptr<AdvantageActorCritic> trainer
    ) : handler(trainer->handler), trainer(trainer) {}

    virtual std::shared_ptr<Economy> setup() {
        std::shared_ptr<NeuralEconomy> economy = get_economy();

        auto person1 = UtilMaxer::init(
            economy,
            Eigen::Array2d(10.0, 10.0),
            20.0,
            std::make_shared<CES>(1.0, Eigen::Array3d(0.5, 0.5, 0.5), 1.3),
            0.8,
            std::make_shared<NeuralPersonDecisionMaker>(economy->handler)
        );
        auto person2 = UtilMaxer::init(
            economy,
            Eigen::Array2d(10.0, 10.0),
            20.0,
            std::make_shared<CES>(1.0, Eigen::Array3d(0.2, 0.6, 0.4), 1.3),
            0.9,
            std::make_shared<NeuralPersonDecisionMaker>(economy->handler)
        );

        auto firm = ProfitMaxer::init(
            economy,
            std::vector<std::shared_ptr<Agent>>({person1}),
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

    std::shared_ptr<DecisionNetHandler> handler;
    std::shared_ptr<AdvantageActorCritic> trainer;

protected:
    std::shared_ptr<NeuralEconomy> get_economy() {
        std::shared_ptr<NeuralEconomy> economy;
        // if handler not yet initialized, initialize as default handler
        if (handler == nullptr) {
            economy = NeuralEconomy::init({"bread", "capital"}, 3);
            handler = economy->handler;
        }
        else {
            economy = NeuralEconomy::init({"bread", "capital"}, 3, handler);
        }
        // also make sure trainer is initialized
        if (trainer == nullptr) {
            trainer = std::make_shared<AdvantageActorCritic>(handler);
        }
        return economy;
    }
};

} // namespace neural



#endif