#ifndef VECTOSCALAR_H
#define VECTOSCALAR_H

#include <Eigen/Dense>

using Vec = Eigen::ArrayXd;


// returns max value in values with starting index given as a pointer
// starting index will be modified in place to be the index of the max value
double min(const Vec& values, unsigned int length, unsigned int* startIdx);


class VecToScalar {
    // base class meant to store parameters for a real-valued function & its derivatives
public:
    // f is the function managed by VecToScalar, it is scalar-valued function of vector of doubles
    virtual double f(const Vec& quantities);
    // df is the derivative of f with respect to the idx'th input quantity
    virtual double df(const Vec& quantities, unsigned int idx);
    unsigned int getNumInputs();
protected:
    void check_no_length_change(const Vec& candidate);
    unsigned int numInputs;
};


class Linear : public VecToScalar {
    // Perfect substitutes
public:
    Linear(const Vec& productivities);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_productivities(const Vec& newProductivities);
protected:
    Vec productivities;
};


class CobbDouglas : public VecToScalar {
public:
    CobbDouglas(double tfp, const Vec& elasticities);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_tfp(double newTfp);
    virtual void set_elasticities(const Vec& newElasticities);
protected:
    double tfp;
    Vec elasticities;
};


class CobbDouglasCRS : public CobbDouglas {
public:
    CobbDouglasCRS(double tfp, const Vec& elasticities);
    void set_elasticities(const Vec& newElasticities);
};


class StoneGeary : public CobbDouglas {
public:
    StoneGeary(double tfp, const Vec& elasticities, const Vec& thresholdParams);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_thresholdParams(const Vec& newThresholdParams);
protected:
    Vec thresholdParams;
};


class Leontief : public VecToScalar {
    // Perfect compliments
public:
    Leontief(const Vec& productivities);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_productivities(const Vec& newProductivities);
protected:
    Vec productivities;
};


class CES : public VecToScalar {
    // Constant elasticity of substitution
public:
    CES(double tfp, const Vec& shareParams, double elasticityOfSubstitution);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_shareParams(const Vec& newShareParams);
    void set_substitutionParam(double newElasticityOfSubstitution);
protected:
    double tfp;
    Vec shareParams;
    double substitutionParam;
    double get_inner_sum(const Vec& quantities);
};


class ProfitFunc : public VecToScalar {
    // Encapsulates another VecToScalar to return the profit for different levels of production
public:
    ProfitFunc(double price, const Vec& factorPrices, const VecToScalar& prodFunc);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_price(double newPrice);
    void set_factorPrices(const Vec& newFactorPrices);
    void set_prodFunc(VecToScalar newProdFunc);
protected:
    double price;
    VecToScalar prodFunc;
    Linear costFunc;
};

#endif
