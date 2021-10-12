#include <iostream>
#include "solve.h"

VarSet::VarSet(const std::string& name, int numVars) : ifopt::VariableSet(numVars, name), vars(Eigen::VectorXd::Zero(numVars)), bounds(std::vector<ifopt::Bounds>(numVars, ifopt::NoBound)) {}

VarSet::VarSet(const std::string& name, int numVars, Eigen::VectorXd initVals) : ifopt::VariableSet(numVars, name), vars(initVals), bounds(std::vector<ifopt::Bounds>(numVars, ifopt::NoBound)) {}

VarSet::VarSet(const std::string& name, int numVars, Eigen::VectorXd initVals, std::vector<ifopt::Bounds> bounds) : ifopt::VariableSet(numVars, name), vars(initVals), bounds(bounds) {}


ConstrSet::ConstrSet(
    const std::string& name,
    const std::string& varName,
    int numConstrs,
    std::shared_ptr<VecToVec> constrFunc,
    std::vector<ifopt::Bounds> bounds
) : ifopt::ConstraintSet(numConstrs, name), varName(varName), constrFunc(constrFunc), bounds(bounds) {
    assert(constrFunc->get_numOutputs() == numConstrs);
    assert(bounds.size() == numConstrs);
}

Eigen::VectorXd ConstrSet::GetValues() const {
    Eigen::ArrayXd x = GetVariables()->GetComponent(varName)->GetValues().array();
    return constrFunc->f(x).matrix();
}

void ConstrSet::FillJacobianBlock(std::string var_set, Jacobian& jac_block) const {
    if (var_set == varName) {
        Eigen::ArrayXd x = GetVariables()->GetComponent(varName)->GetValues().array();
        for (unsigned int i = 0; i < constrFunc->get_numOutputs(); i++) {
            for (unsigned int j = 0; j < constrFunc->get_numInputs(); j++) {
                jac_block.coeffRef(i, j) = constrFunc->df(x, i, j);
            }
        }
    }
}

void ConstrSet::FillJacobianBlock(Jacobian& jac_block) const {
    return FillJacobianBlock(varName, jac_block);
}


Objective::Objective(const std::string& name, const std::string& varName, std::shared_ptr<VecToScalar> objectiveFunc) : ifopt::CostTerm(name), varName(varName), objectiveFunc(objectiveFunc) {}

double Objective::GetCost() const {
    Eigen::ArrayXd x = GetVariables()->GetComponent(varName)->GetValues().array();
    // notice we return the negative objectiveFunc since we're trying to maximize objective
    return -objectiveFunc->f(x);
}

void Objective::FillJacobianBlock(std::string var_set, Jacobian& jac) const {
    if (var_set == varName) {
        Eigen::ArrayXd x = GetVariables()->GetComponent(varName)->GetValues().array();
        for (unsigned int j = 0; j < objectiveFunc->get_numInputs(); j++) {
            jac.coeffRef(0, j) = -objectiveFunc->df(x, j);
        }
    }
}

void Objective::FillJacobianBlock(Jacobian& jac) const {
    return FillJacobianBlock(varName, jac);
}


void configure_to_default_solver(std::shared_ptr<ifopt::IpoptSolver> solver) {
    // use MUMPS as linear solver
    // if you have the HSL solvers, you should use those instead
    solver->SetOption("linear_solver", "mumps");
    // require jacobians to be pre-provided
    solver->SetOption("jacobian_approximation", "exact");
    solver->SetOption("print_level", 1);
}


Problem::Problem(std::shared_ptr<VarSet> varSet, std::shared_ptr<ConstrSet> constrSet, std::shared_ptr<Objective> objective) {
    problem.AddVariableSet(varSet);
    problem.AddConstraintSet(constrSet);
    problem.AddCostSet(objective);
    configure_to_default_solver(solver);
}

Eigen::ArrayXd Problem::solve() {
    solver->Solve(problem);
    return problem.GetOptVariables()->GetValues().array();
}

void Problem::changeSolver(std::shared_ptr<ifopt::IpoptSolver> newSolver) {
    solver = newSolver;
}
