#ifndef SCENARIO_H
#define SCENARIO_H

#include <memory>
#include "base.h"

struct Scenario {
    /**
     * This is simply a container that generates an Economy according to some user-defined specification.
     * 
     * It's not necessary to use this to create a simulation scenario,
     * but it's helpful when the same scenario must be recreated many times,
     * e.g. when training the neural nets in a neural::NeuralEconomy
     */
    Scenario() {}
    virtual ~Scenario() {}
    virtual std::shared_ptr<Economy> setup() = 0;
};

#endif