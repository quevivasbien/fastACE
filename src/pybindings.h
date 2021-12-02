#ifndef PYBINDINGS_H
#define PYBINDINGS_H

#include "constants.h"
#include "neuralScenarios.h"


extern "C" {
    auto* get_config();

    neural::CustomScenarioParams create_scenario_params(
        unsigned int numPeople,
        unsigned int numFirms
    );

    neural::TrainingParams create_training_params(
        unsigned int numEpisodes,
        unsigned int episodeLength,
        unsigned int updateEveryNEpisodes,
        unsigned int checkpointEveryNEpisodes
    );

    void train(
        float* output,
        neural::CustomScenarioParams scenarioParams,
        neural::TrainingParams trainingParams
    );
}

#endif