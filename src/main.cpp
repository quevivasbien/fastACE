#include <memory>
#include "neuralScenarios.h"


int main() {

    auto scenario = neural::SimpleScenario();
    auto economy = scenario.setup();

    economy->time_step();
    scenario.trainer->train_on_episode();

    return 0;
}
