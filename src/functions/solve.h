#ifndef SOLVE_H
#define SOLVE_H

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>


using Vec = Eigen::ArrayXd;

// Functor definition is based on MattKelly's answer to:
// https://stackoverflow.com/questions/18509228/how-to-use-the-eigen-unsupported-levenberg-marquardt-implementation
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor {
    // Information that tells the caller the numeric type (eg. double) and size (input / output dim)
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    // Tell the caller the matrix sizes associated with the input, output, and jacobian
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;
    // local copy of number of inputs
    int m_inputs, m_values;
    // define constructors
    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}
    // define methods for users to get input and output dimensions
    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
};

// example functor for maximization
struct HimmelblauFunctor : Functor<double> {
    // define constructor
    HimmelblauFunctor() : Functor<double>(2, 2) {}
    // implementation of objective function
    int operator() (const Eigen::VectorXd &z, Eigen::VectorXd &fvec) const {
        double x = z(0);
        double y = z(1);
        /*
         * Evaluate Himmelblau's function.
         * Important: LevenbergMarquardt is designed to work with objective functions that are a sum
         * of squared terms. The algorithm takes this into account: do not do it yourself.
         * In other words: objFun = sum(fvec(i)^2)
         */
        fvec(0) = x * x + y - 11;
        fvec(1) = x + y * y - 7;
        return 0;
    }
};

void testHimmelblau(Eigen::VectorXd zInit) {
    std::cout << "Testing Himmelblau function..." << std::endl;
    std::cout << "zInit: " << zInit.transpose() << std::endl;
    HimmelblauFunctor functor;
    // define a function numDiff which gives us the jacobian
    Eigen::NumericalDiff<HimmelblauFunctor> numDiff(functor);
    // define lm, which is the actual solver
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<HimmelblauFunctor>, double> lm(numDiff);
    // change some parameters
    lm.parameters.maxfev = 1000;
    lm.parameters.xtol = 1.0e-10;
    // solve
    Eigen::VectorXd z = zInit;
    int ret = lm.minimize(z);
    std::cout << "iter count: " << lm.iter << std::endl;
    std::cout << "return status: " << ret << std::endl;
    std::cout << "zSolver: " << z.transpose() << std::endl;
}


#endif
