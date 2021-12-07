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

    neural::TrainingParams create_training_params();

    void run(
        neural::CustomScenarioParams scenarioParams,
        neural::TrainingParams trainingParams
    );

    void train(
        float* output,
        neural::CustomScenarioParams scenarioParams,
        neural::TrainingParams trainingParams,
        bool fromPretrained
    );
}

#endif