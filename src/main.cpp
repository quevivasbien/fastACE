#include <memory>
#include <string>
#include "neuralScenarios.h"


int main(int argc, char *argv[]) {

    int numPersons = (argc > 1) ? std::stoul(argv[1]) : 20;
    int numFirms = (argc > 2) ? std::stoul(argv[2]) : 4;

    int numEpisodes = (argc > 3) ? std::stoul(argv[3]) : 10;
    int episodeLength = (argc > 4) ? std::stoul(argv[4]) : 10;

    auto scenario = std::make_shared<neural::VariablePopulationScenario>(numPersons, numFirms);

    neural::train(scenario, numEpisodes, episodeLength);

    return 0;
}
