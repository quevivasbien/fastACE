#ifndef SOLVE_H
#define SOLVE_H

#include <memory>
#include <string>
#include <vector>
#include <assert.h>
#include <ifopt/variable_set.h>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include "vecToVec.h"
#include "vecToScalar.h"

using Jacobian = Eigen::SparseMatrix<double, Eigen::RowMajor>;


class VarSet : public ifopt::VariableSet {
    // a friendlier wrapper for variable sets
public:
    VarSet(const std::string& name, int numVars);
    VarSet(const std::string& name, int numVars, Eigen::VectorXd initVals);
    VarSet(const std::string& name, int numVars, Eigen::VectorXd initVals, std::vector<ifopt::Bounds> bounds);

    void SetVariables(const Eigen::VectorXd& x) override {
        assert(x.size() == GetRows());
        vars = x;
    }

    Eigen::VectorXd GetValues() const override {
        return vars;
    }

    std::vector<ifopt::Bounds> GetBounds() const override {
        return bounds;
    };

private:
    Eigen::VectorXd vars;
    std::vector<ifopt::Bounds> bounds;
};


class ConstrSet : public ifopt::ConstraintSet {
    // wrapper for constraint sets
public:
    ConstrSet(
        const std::string& name,  // the name for this constraint
        const std::string& varName,  // the name of the varSet this constraint applies to
        int numConstrs,  // number of constraints
        std::shared_ptr<VecToVec> constrFunc,  // vector function to apply to inputs
        std::vector<ifopt::Bounds> bounds  // bounds on output of constrFunc
    );

    Eigen::VectorXd GetValues() const override;

    std::vector<ifopt::Bounds> GetBounds() const override {
        return bounds;
    }

    void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

    void FillJacobianBlock(Jacobian& jac_block) const;

private:
    std::string varName;
    std::shared_ptr<VecToVec> constrFunc;
    std::vector<ifopt::Bounds> bounds;
};


class Objective : public ifopt::CostTerm {
    // wrapper for cost terms.
    // note that this actually is framed in terms of a function we want to _maximize_, not minimize
    // since in economics that's typically what we're trying to do
public:
    Objective(const std::string& name, const std::string& varName, std::shared_ptr<VecToScalar> objectiveFunc);

    double GetCost() const override;

    void FillJacobianBlock(std::string var_set, Jacobian& jac) const override;

    void FillJacobianBlock(Jacobian& jac) const;

private:
    std::string varName;
    std::shared_ptr<VecToScalar> objectiveFunc;
};


#endif
