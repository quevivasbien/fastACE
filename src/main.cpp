#include <memory>
#include "neuralScenarios.h"


int main() {

    auto scenario = std::make_shared<neural::VariablePopulationScenario>(100, 5);

    neural::train(scenario, 10, 5);

    return 0;
}
