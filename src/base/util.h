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

inline std::default_random_engine get_rng() {
    unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::default_random_engine(seed);
}

// helper function used for filtering offers by availability
template <typename T>
std::vector<std::shared_ptr<const T>> filter_available(
    std::shared_ptr<Agent> requester,
    const std::vector<std::shared_ptr<const T>>& offers,
    std::default_random_engine rng
) {
    std::vector<std::shared_ptr<const T>> availOffers;
    availOffers.reserve(offers.size());
    for (auto offer : offers) {
        if (offer->is_available() && (offer->offerer != requester)) {
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
void flush(std::vector<std::shared_ptr<T>>& offers) {
    // figure out which offers are no longer available
    std::vector<unsigned int> idxs;
    for (unsigned int i = 0; i < offers.size(); i++) {
        if (!offers[i]->is_available()) {
            idxs.push_back(i);
        }
    }
    // remove those offers
    for (auto i : idxs) {
        offers[i] = offers.back();
        offers.pop_back();
    }
}


template <typename ... Args>
void print(Args&& ... args) {
    if (constants::verbose > 0) {
        for (auto arg : {args...}) {
    		std::cout << arg << ' ';
    	}
    	std::cout << '\n';
    }
}

template <typename ... Args>
void print(unsigned int priority, Args&& ... args) {
    if (constants::verbose >= priority) {
        print({args...});
    }
}


template <typename T>
inline void print_status(T* origin, std::string status) {
    if (constants::verbose >= 2) {
        std::cout << origin
            << " (" << origin->get_typename() << ") : "
            << status << '\n';
    }
}


template <typename T>
inline void print_status(std::shared_ptr<T> origin, std::string status) {
    if (constants::verbose >= 2) {
        std::cout << origin
            << " (" << origin->get_typename() << ") : "
            << status << '\n';
    }
}


#endif
