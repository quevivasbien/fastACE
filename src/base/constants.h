#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <thread>

namespace constants {
    const unsigned int verbose = 1;
    const double defaultPrice = 1.0;
    const double priceMultiplier = 1.1;
    const double defaultLaborBudget = 0.5;
    const double defaultWage = 1.0;
    const double laborIncrement = 0.25;
    const unsigned int heat = 5;
    const double eps = 1e-8;
    const double largeNumber = 1e8;
    const bool multithreaded = true;
    const unsigned int numThreads = std::thread::hardware_concurrency();
}

#endif
