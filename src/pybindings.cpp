#include <cstddef>
#include <vector>
#include <iostream>
#include "neuralScenarios.h"

extern "C" {
    void train(
        float* output,
        unsigned int numPersons,
        unsigned int numFirms,
        unsigned int numEpisodes,
        unsigned int episodeLength
    );
}

void train(
    float* output,
    unsigned int numPersons,
    unsigned int numFirms,
    unsigned int numEpisodes,
    unsigned int episodeLength
) {
    auto scenario = std::make_shared<neural::VariablePopulationScenario>(numPersons, numFirms);
    std::vector<float> losses = neural::train(scenario, numEpisodes, episodeLength);
    std::copy(losses.begin(), losses.end(), output);
}
