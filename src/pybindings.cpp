#include <vector>
#include <iostream>
#include <memory>
#include "pybindings.h"


auto* get_config() {
    return &constants::config;
}

neural::CustomScenarioParams create_scenario_params(
    unsigned int numPeople,
    unsigned int numFirms
) {
    return neural::CustomScenarioParams(numPeople, numFirms);
}

neural::TrainingParams create_training_params(
) {
    return neural::TrainingParams();
}


void run(
    neural::CustomScenarioParams scenarioParams,
    neural::TrainingParams trainingParams
) {
    std::shared_ptr<neural::CustomScenario> scenario = neural::create_scenario(scenarioParams, trainingParams);
    scenario->handler->load_models();
    auto economy = std::static_pointer_cast<neural::NeuralEconomy>(scenario->setup());
    for (unsigned int t = 0; t < trainingParams.episodeLength; t++) {
        economy->time_step_no_grad();
    }
}


void train(
    float* output,
    neural::CustomScenarioParams scenarioParams,
    neural::TrainingParams trainingParams,
    bool fromPretrained
) {
    std::vector<float> losses = (
        (!fromPretrained) ?
        neural::train(
            scenarioParams,
            trainingParams
        )
        :
        neural::train_from_pretrained(
            scenarioParams,
            trainingParams
        )
    );
    
    std::copy(losses.begin(), losses.end(), output);
}
