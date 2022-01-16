#ifndef PYBINDINGS_H
#define PYBINDINGS_H

#include "constants.h"
#include "neuralScenarios.h"


extern "C" {
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
        double* output,
        const neural::CustomScenarioParams* scenarioParams,
        neural::TrainingParams* trainingParams,
        bool fromPretrained,
        double perturbationSize = 0.0
    );
}

#endif