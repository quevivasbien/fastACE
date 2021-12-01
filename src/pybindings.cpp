#include <vector>
#include <iostream>
#include "pybindings.h"


neural::CustomScenarioParams create_scenario_params(
    unsigned int numPeople,
    unsigned int numFirms
) {
    return neural::CustomScenarioParams(numPeople, numFirms);
}

neural::TrainingParams create_training_params(
    unsigned int numEpisodes,
    unsigned int episodeLength,
    unsigned int updateEveryNEpisodes,
    unsigned int checkpointEveryNEpisodes
) {
    return neural::TrainingParams(
        numEpisodes,
        episodeLength,
        updateEveryNEpisodes,
        checkpointEveryNEpisodes
    );
}


void train(
    float* output,
    neural::CustomScenarioParams scenarioParams,
    neural::TrainingParams trainingParams
) {
    std::vector<float> losses = neural::train(
        scenarioParams,
        trainingParams
    );
    
    std::copy(losses.begin(), losses.end(), output);
}
