#ifndef SOLVE_H
#define SOLVE_H

#include <iostream>
#include <string>
#include <ifopt/variable_set.h>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>

namespace ifopt {
using Eigen::Vector2d;


class ExVariables : public VariableSet {
public:
  // Every variable set has a name, here "var_set1". this allows the constraints
  // and costs to define values and Jacobians specifically w.r.t this variable set.
  ExVariables() : ExVariables("var_set1") {};
  ExVariables(const std::string& name) : VariableSet(2, name)
  {
    // the initial values where the NLP starts iterating from
    x0_ = 3.5;
    x1_ = 1.5;
  }

  // Here is where you can transform the Eigen::Vector into whatever
  // internal representation of your variables you have (here two doubles, but
  // can also be complex classes such as splines, etc..
  void SetVariables(const VectorXd& x) override
  {
    x0_ = x(0);
    x1_ = x(1);
  };

  // Here is the reverse transformation from the internal representation to
  // to the Eigen::Vector
  VectorXd GetValues() const override
  {
    return Vector2d(x0_, x1_);
  };

  // Each variable has an upper and lower bound set here
  VecBound GetBounds() const override
  {
    VecBound bounds(GetRows());
    bounds.at(0) = Bounds(-1.0, 1.0);
    bounds.at(1) = NoBound;
    return bounds;
  }

private:
  double x0_, x1_;
};

}


#endif
