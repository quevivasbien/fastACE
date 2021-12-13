#include <cmath>
#include "util.h"

namespace util {

std::default_random_engine get_rng() {
    unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::default_random_engine(seed);
}

std::vector<unsigned int> get_indices_for_multithreading(unsigned int numAgents) {
    unsigned int agentsPerThread = numAgents / constants::numThreads;
    unsigned int extras = numAgents % constants::numThreads;
    std::vector<unsigned int> indices(constants::numThreads + 1);
    indices[0] = 0;
    for (unsigned int i = 1; i <= constants::numThreads; i++) {
        indices[i] = indices[i-1] + agentsPerThread + (i <= extras);
    }
    return indices;
}

void pprint(unsigned int priority, const std::string& message) {
    if (constants::verbose >= priority) {
        std::cout << message << std::endl;
    }
}

void print(const std::string& message) {
    pprint(1, message);
}

void pprint_time_elasped(
    unsigned int priority,
    std::chrono::time_point<std::chrono::system_clock> start_time,
    std::chrono::time_point<std::chrono::system_clock> end_time
) {
    std::chrono::duration<double> elasped_seconds = end_time - start_time;
    if (constants::verbose >= priority) {
        std::cout << elasped_seconds.count() << "s" << std::endl;
    }
}

std::string format_sci_notation(double x) {
    double pow_ = floor(log10(x));
    double head = x / pow(10.0, pow_);
    return std::to_string(head) + 'e' + std::to_string(static_cast<int>(pow_));
}

} // namespace util
