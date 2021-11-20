#include <memory>
#include "neuralScenarios.h"


int main() {

    auto scenario = neural::SimpleScenario();
    auto economy = scenario.setup();

    for (int t = 0; t < 10; t++) {
        economy->time_step();
    }

    return 0;
}
