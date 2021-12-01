#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <thread>

namespace constants {
    const unsigned int verbose = 1;
    const double eps = 1e-8;
    const double largeNumber = 1e8;
    const bool multithreaded = true;
    const unsigned int numThreads = std::thread::hardware_concurrency();

    // Typically constants will be accessed via the config object,
    // which the user can change, allowing the "constants" to technically be non-constant at runtime
    struct {
        unsigned int verbose;
        double eps;
        double largeNumber;
        bool multithreaded;
        unsigned int numThreads;
    } config = {verbose, eps, largeNumber, multithreaded, numThreads};
}

#endif
