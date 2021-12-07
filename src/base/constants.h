#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <thread>

namespace constants {
    // Note: if you compile with low verbosity, the compiler may remove any branches that check for higher verbosity,
    // So if you want to be able to change verbosity after compilation, you need to set here the highest verbosity you plan on using
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
