#ifndef VEC_TO_SCALAR_H
#define VEC_TO_SCALAR_H

#include <memory>
#include <Eigen/Dense>


// returns max value in values with starting index given as a pointer
// starting index will be modified in place to be the index of the max value
double min(const Eigen::ArrayXd& values, unsigned int length, unsigned int* startIdx);


class VecToScalar {
public:
    // base class meant to store parameters for a real-valued function & its derivatives
    virtual ~VecToScalar() {}
    VecToScalar(unsigned int numInputs) : numInputs(numInputs) {}
    // f is the function managed by VecToScalar, it is scalar-valued function of vector of doubles
    virtual double f(const Eigen::ArrayXd& quantities) const = 0;
    // df is the derivative of f with respect to the idx'th input quantity
    virtual double df(const Eigen::ArrayXd& quantities, unsigned int idx) const = 0;

    void check_no_length_change(const Eigen::ArrayXd& candidate) const;
    unsigned int numInputs;
};


class Linear : public VecToScalar {
public:
    // Perfect substitutes
    Linear(unsigned int numInputs);
    Linear(const Eigen::ArrayXd& productivities);
    virtual double f(const Eigen::ArrayXd& quantities) const override;
    virtual double df(const Eigen::ArrayXd& quantities, unsigned int idx) const override;

    Eigen::ArrayXd productivities;
};


class CobbDouglas : public VecToScalar {
public:
    CobbDouglas(unsigned int numInputs);
    CobbDouglas(double tfp, const Eigen::ArrayXd& elasticities);
    virtual double f(const Eigen::ArrayXd& quantities) const;
    virtual double df(const Eigen::ArrayXd& quantities, unsigned int idx) const;

    double tfp;
    Eigen::ArrayXd elasticities;
};


class CobbDouglasCRS : public CobbDouglas {
public:
    CobbDouglasCRS(double tfp, const Eigen::ArrayXd& elasticities);
};


class StoneGeary : public CobbDouglas {
public:
    StoneGeary(double tfp, const Eigen::ArrayXd& elasticities, const Eigen::ArrayXd& thresholdParams);
    virtual double f(const Eigen::ArrayXd& quantities) const;
    virtual double df(const Eigen::ArrayXd& quantities, unsigned int idx) const;

    Eigen::ArrayXd thresholdParams;
};


class Leontief : public VecToScalar {
public:
    // Perfect compliments
    Leontief(const Eigen::ArrayXd& productivities);
    virtual double f(const Eigen::ArrayXd& quantities) const;
    virtual double df(const Eigen::ArrayXd& quantities, unsigned int idx) const;

    Eigen::ArrayXd productivities;
};


class CES : public VecToScalar {
public:
    // Constant elasticity of substitution
    // elast = 1 -> CobbDouglas
    // elast = infty -> Linear
    // elast = 0 -> Leontief
    CES(double tfp, const Eigen::ArrayXd& shareParams, double elasticityOfSubstitution);
    virtual double f(const Eigen::ArrayXd& quantities) const;
    virtual double df(const Eigen::ArrayXd& quantities, unsigned int idx) const;

    double tfp;
    Eigen::ArrayXd shareParams;
    double substitutionParam;
    double get_inner_sum(const Eigen::ArrayXd& quantities) const;
};


class ProfitFunc : public VecToScalar {
public:
    // Encapsulates another VecToScalar to return the profit for different levels of production
    ProfitFunc(double price, const Eigen::ArrayXd& factorPrices, std::shared_ptr<VecToScalar> prodFunc);
    virtual double f(const Eigen::ArrayXd& quantities) const;
    virtual double df(const Eigen::ArrayXd& quantities, unsigned int idx) const;

    double price;
    std::shared_ptr<VecToScalar> prodFunc;
    Linear costFunc;
};

#endif
