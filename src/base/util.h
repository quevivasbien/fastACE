#ifndef UTIL_H
#define UTIL_H

#include <utility>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include "constants.h"

class Agent;

namespace util {

std::default_random_engine get_rng();

// helper function used for filtering offers by availability
template <typename T>
std::vector<std::weak_ptr<const T>> filter_available(
    std::shared_ptr<Agent> requester,
    const std::vector<std::weak_ptr<const T>>& offers,
    std::default_random_engine rng
) {
    std::vector<std::weak_ptr<const T>> availOffers;
    availOffers.reserve(offers.size());
    for (auto offer_ : offers) {
        auto offer = offer_.lock();
        if (offer != nullptr && offer->is_available() && (offer->offerer.lock() != requester)) {
            availOffers.push_back(offer);
        }
    }
    std::shuffle(std::begin(availOffers), std::end(availOffers), rng);
    return availOffers;
}

// a helper function template for instantiating Agent objects
// should be included as a friend function in any class that inherits from Agent
template <typename T, typename ... Args>
std::shared_ptr<T> create(Args&& ... args) {
	std::shared_ptr<T> agent = std::shared_ptr<T>(new T(std::forward<Args>(args) ...));
    agent->economy->add_agent(agent);
    return agent;
}


// helper for getting rid of unavailable offers
template <typename T>
void flush(
    std::vector<std::weak_ptr<T>>& offers
) {
    offers.erase(
        std::remove_if(
            offers.begin(),
            offers.end(),
            [](std::weak_ptr<T> offer_) {
                auto offer = offer_.lock();
                return (offer == nullptr || !offer->is_available());
            }
        ),
        offers.end()
    );
}

template <typename T>
void flush(
    std::vector<std::shared_ptr<T>>& offers
) {
    offers.erase(
        std::remove_if(
            offers.begin(),
            offers.end(),
            [](std::shared_ptr<T> offer) { return !offer->is_available(); }
        ),
        offers.end()
    );
}


// helper for dividing up agents to be operated on by multiple threads
std::vector<unsigned int> get_indices_for_multithreading(unsigned int numAgents);


template <typename T>
T make_positive(T x) {
    if (x <= 0) {
        return constants::eps;
    }
    else {
        return x;
    }
}

template <typename T>
T make_nonnegative(T x) {
    if (x < 0) {
        return 0.0;
    }
    else {
        return x;
    }
}



// FUNCTIONS FOR PRINTING BASED ON VALUE OF constants::verbose

// pprint = priority print: prints message if constants::verbose >= priority
void pprint(unsigned int priority, const std::string& message);

template <typename ... Args>
void pprint(unsigned int priority, Args&& ... args) {
    if (constants::verbose >= priority) {
        for (auto arg : {args...}) {
    		std::cout << arg << ' ';
    	}
    	std::cout << '\n';
    }
}

// print if verbose > 0
void print(const std::string& message);

template <typename ... Args>
void print(Args&& ... args) {
    pprint(1, {args ...});
}


template <typename T>
void pprint_status(
    unsigned int priority,
    T* origin,
    const std::string& status
) {
    if (constants::verbose >= priority) {
        std::cout << origin
            << " (" << origin->get_typename() << ") : "
            << status << '\n';
    }
}

template <typename T>
void pprint_status(
    unsigned int priority,
    std::shared_ptr<T> origin,
    const std::string& status
) {
    if (constants::verbose >= priority) {
        std::cout << origin
            << " (" << origin->get_typename() << ") : "
            << status << '\n';
    }
}

// print status if verbose >= 2
template <typename T>
void print_status(T* origin, const std::string& status) {
    pprint_status(3, origin, status);
}


template <typename T>
void print_status(std::shared_ptr<T> origin, const std::string& status) {
    pprint_status(3, origin, status);
}


void pprint_time_elasped(
    unsigned int priority,
    std::chrono::time_point<std::chrono::system_clock> start_time,
    std::chrono::time_point<std::chrono::system_clock> end_time
);

std::string format_sci_notation(double x);

} // namespace util

#endif
