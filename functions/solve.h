#ifndef SOLVE_H
#define SOLVE_H

#include <vector>
#include <functional>
#include "linalg.h"

using Vec = std::vector<double>;

Vec newtons_method(
    std::function<Vec(Vec)> f,
    std::function<Vec(Vec)> grad,
    std::function<Matrix(Vec)> hess,
    Vec x0
);

Vec newtons_method(
    std::function<Vec(Vec)> f,
    std::function<Vec(Vec)> grad,
    std::function<Matrix(Vec)> hess,
    Vec x0,
    double tol
);

Vec gradient_descent(
    std::function<Vec(Vec)> f,
    Vec x0,
    double tol
);

Vec gradient_descent(
    std::function<Vec(Vec)> f,
    Vec x0,
    double tol
);

#endif
