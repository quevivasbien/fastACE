#include "neuralScenarios.h"

namespace neural {

NeuralScenario::NeuralScenario() : handler(nullptr), trainer(nullptr) {}

NeuralScenario::NeuralScenario(
    std::shared_ptr<AdvantageActorCritic> trainer
) : handler(trainer->handler), trainer(trainer) {}

std::shared_ptr<NeuralEconomy> NeuralScenario::get_economy(
    std::vector<std::string> goods
) {
    std::shared_ptr<NeuralEconomy> economy;
    // if handler not yet initialized, initialize as default handler
    if (handler == nullptr) {
        economy = NeuralEconomy::init(goods);
        handler = economy->handler.lock();
    }
    else {
        economy = NeuralEconomy::init(goods, handler);
        handler->reset(economy);
    }
    // also make sure trainer is initialized
    if (trainer == nullptr) {
        trainer = std::make_shared<AdvantageActorCritic>(handler);
    }
    return economy;
}


SimpleScenario::SimpleScenario() : NeuralScenario() {}
SimpleScenario::SimpleScenario(
    std::shared_ptr<AdvantageActorCritic> trainer
) : NeuralScenario(trainer) {}

std::shared_ptr<Economy> SimpleScenario::setup() {
    std::shared_ptr<NeuralEconomy> economy = get_economy({"bread", "capital"});

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


CustomScenario::CustomScenario(
    unsigned int numPeople,
    unsigned int numFirms
) : NeuralScenario(), params(CustomScenarioParams(numPeople, numFirms)) {}

CustomScenario::CustomScenario(
    std::shared_ptr<AdvantageActorCritic> trainer,
    unsigned int numPeople,
    unsigned int numFirms
) : NeuralScenario(trainer), params(CustomScenarioParams(numPeople, numFirms)) {}

CustomScenario::CustomScenario(
    std::shared_ptr<AdvantageActorCritic> trainer,
    CustomScenarioParams params
) : NeuralScenario(trainer), params(params) {}

std::shared_ptr<Economy> CustomScenario::setup() {
    std::shared_ptr<NeuralEconomy> economy = get_economy(
        {"bread", "capital"}
    );

    auto rng = economy->get_rng();
    std::normal_distribution<double> randn(0, 1);

    for (unsigned int i = 0; i < params.numPeople; i++) {
        ;
        UtilMaxer::init(
            economy,
            Eigen::Array2d(
                util::make_nonnegative(params.good1_mu + params.good1_sigma * randn(rng)),
                util::make_nonnegative(params.good2_mu + params.good2_sigma * randn(rng))
            ),
            util::make_nonnegative(params.money_mu + params.money_sigma * randn(rng)),
            std::make_shared<CES>(
                1.0,
                Eigen::Array3d(
                    params.labor_share_mu + params.labor_share_sigma * randn(rng),
                    params.good1_share_mu + params.good1_share_sigma * randn(rng),
                    params.good2_share_mu + params.good2_share_sigma * randn(rng)
                ),
                util::make_positive(params.elasticity_mu + params.elasticity_sigma * randn(rng))
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
                util::make_nonnegative(params.firm_good1_mu + params.firm_good2_sigma * randn(rng)),
                util::make_nonnegative(params.firm_good2_mu + params.firm_good2_sigma * randn(rng))
            ),
            util::make_nonnegative(params.firm_money_mu + params.firm_money_sigma * randn(rng)),
            create_CES_VecToVec(
                std::vector<double>({
                    util::make_nonnegative(params.firm_tfp1_mu + params.firm_tfp1_sigma * randn(rng)),
                    util::make_nonnegative(params.firm_tfp2_mu + params.firm_tfp2_sigma * randn(rng))
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
                    util::make_positive(params.firm_elasticity1_mu + params.firm_elasticity1_sigma * randn(rng)),
                    util::make_positive(params.firm_elasticity2_mu + params.firm_elasticity2_sigma * randn(rng))
                })
            ),
            std::make_shared<NeuralFirmDecisionMaker>(economy->handler)
        );
    }

    return economy;
}

std::shared_ptr<NeuralEconomy> CustomScenario::setup_dummy() {
    return NeuralEconomy::init_dummy(2);
}


std::shared_ptr<CustomScenario> create_scenario(
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
        trainingParams.multiplierForLRDecay,
        trainingParams.reverseAnnealingPeriod
    );

    return std::make_shared<CustomScenario>(trainer, scenarioParams);
}


std::vector<double> train(
    const std::shared_ptr<NeuralScenario>& scenario,
    const TrainingParams& params
) {
    // note: this implementation will ignore all TrainingParam members except the first four:
    // numEpisodes, episodeLength, updateEveryNEpisodes, checkPointEveryNEpisodes
    // other params are taken from scenario->trainer

    // If you want to specify more options, use the overloaded train function that takes as args
    // a CustomScenarioParams struct and a TrainingParams struct
    auto start = std::chrono::system_clock::now();
    std::shared_ptr<Economy> economy;
    std::vector<double> losses(params.numEpisodes);
    for (unsigned int i = 0; i < params.numEpisodes; i++) {
        economy = scenario->setup();

        auto step_time_start = std::chrono::system_clock::now();
        for (unsigned int t = 0; t < params.episodeLength; t++) {
            economy->time_step();
        }
        auto step_time_end = std::chrono::system_clock::now();

        double loss = scenario->trainer->train_on_episode();
        auto train_time_end = std::chrono::system_clock::now();

        util::pprint(2, "Time spent time stepping:");
        util::pprint_time_elasped(2, step_time_start, step_time_end);
        util::pprint(2, "Time spent time training:");
        util::pprint_time_elasped(2, step_time_end, train_time_end);

        if (std::isnan(loss)) {
            if (i >= params.checkpointEveryNEpisodes) {
                util::pprint(
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
            double sum = 0.0;
            for (int j = 0; j < params.updateEveryNEpisodes; j++) {
                sum += losses[i - j];
            }
            double avg = sum / params.updateEveryNEpisodes;
            util::print(
                "Episode " + std::to_string(i+1)
                + ": Average loss over past " + std::to_string(params.updateEveryNEpisodes)
                + " episodes = " + util::format_sci_notation(sum / params.updateEveryNEpisodes)
            );
        }
    }
    auto end = std::chrono::system_clock::now();
    util::pprint(1, "Total time:");
    util::pprint_time_elasped(1, start, end);

    return losses;
}

std::vector<double> train(
    const std::shared_ptr<NeuralScenario>& scenario,
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

std::vector<double> train(
    const CustomScenarioParams& scenarioParams,
    const TrainingParams& trainingParams
) {
    auto scenario = create_scenario(scenarioParams, trainingParams);
    return train(scenario, trainingParams);
}

std::vector<double> train_from_pretrained(
    const CustomScenarioParams& scenarioParams,
    const TrainingParams& trainingParams
) {
    auto scenario = create_scenario(scenarioParams, trainingParams);
    scenario->handler->load_models();
    return train(scenario, trainingParams);
}


} // namespace neural