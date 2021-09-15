#include <iostream>
#include <vector>
#include <memory>
#include "economy.h"
#include "persons/utilMaxer.h"

const int time_steps = 10;

int main() {
    CobbDouglas cobbDouglas(1.0, std::vector<double>({0.5, 0.5}));
    std::cout << cobbDouglas.f(std::vector<double>({2.0, 1.0})) << std::endl;
    return 0;
}
