#include <iostream>
#include <vector>
#include <memory>
#include "economy.h"
#include "persons/utilMaximizer.h"

const int time_steps = 10;

int main() {
    Economy economy;
    std::shared_ptr<UtilMaximizer> person = std::make_shared<UtilMaximizer>(&economy);
    economy.add_person(person);
    std::vector<double> vector {1.0, 1.0};
    std::cout << person->u(vector) << std::endl;
    return 0;
}
