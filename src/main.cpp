#include <iostream>
#include <memory>
#include "solve.h"
#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>


int main() {
    // tell ifopt we're working with two variables
    std::shared_ptr<VarSet> varSet = std::make_shared<VarSet>("vars", 2, Eigen::Vector2d(0.1, 0.1));
    // the constraint is 0 <= x0 + x1 <= 1
    std::shared_ptr<VecToVecFromVecToScalar> constrFunc = std::make_shared<VecToVecFromVecToScalar>(std::make_shared<Linear>(Eigen::Array2d::Ones()));
    std::vector<ifopt::Bounds> bounds = {ifopt::Bounds(0,1)};
    std::shared_ptr<ConstrSet> constrSet = std::make_shared<ConstrSet>("constraints", "vars", 1, constrFunc, bounds);
    // objective is x0^0.5 * x1^0.5
    std::shared_ptr<CobbDouglasCRS> objFunc = std::make_shared<CobbDouglasCRS>(1.0, Eigen::Array2d::Ones());
    std::shared_ptr<Objective> obj = std::make_shared<Objective>("objective", "vars", objFunc);

    ifopt::Problem problem;
    problem.AddVariableSet(varSet);
    problem.AddConstraintSet(constrSet);
    problem.AddCostSet(obj);

    ifopt::IpoptSolver solver;
    solver.SetOption("linear_solver", "mumps");
    solver.SetOption("jacobian_approximation", "exact");

    solver.Solve(problem);
    Eigen::VectorXd x = problem.GetOptVariables()->GetValues();
    std::cout << x.transpose() << std::endl;

    return 0;
}
