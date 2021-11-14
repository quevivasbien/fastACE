#ifndef VEC_TO_SCALAR_H
#define VEC_TO_SCALAR_H

#include <memory>
#include <Eigen/Dense>

using Vec = Eigen::ArrayXd;


// returns max value in values with starting index given as a pointer
// starting index will be modified in place to be the index of the max value
double min(const Vec& values, unsigned int length, unsigned int* startIdx);


struct VecToScalar {
    // base class meant to store parameters for a real-valued function & its derivatives
    virtual ~VecToScalar() {}
    VecToScalar(unsigned int numInputs) : numInputs(numInputs) {}
    // f is the function managed by VecToScalar, it is scalar-valued function of vector of doubles
    virtual double f(const Vec& quantities) const = 0;
    // df is the derivative of f with respect to the idx'th input quantity
    virtual double df(const Vec& quantities, unsigned int idx) const = 0;

    void check_no_length_change(const Vec& candidate) const;
    unsigned int numInputs;
};


struct Linear : VecToScalar {
    // Perfect substitutes
    Linear(unsigned int numInputs);
    Linear(const Vec& productivities);
    virtual double f(const Vec& quantities) const override;
    virtual double df(const Vec& quantities, unsigned int idx) const override;

    Vec productivities;
};


struct CobbDouglas : VecToScalar {
    CobbDouglas(unsigned int numInputs);
    CobbDouglas(double tfp, const Vec& elasticities);
    virtual double f(const Vec& quantities) const;
    virtual double df(const Vec& quantities, unsigned int idx) const;

    double tfp;
    Vec elasticities;
};


struct CobbDouglasCRS : CobbDouglas {
    CobbDouglasCRS(double tfp, const Vec& elasticities);
};


struct StoneGeary : CobbDouglas {
    StoneGeary(double tfp, const Vec& elasticities, const Vec& thresholdParams);
    virtual double f(const Vec& quantities) const;
    virtual double df(const Vec& quantities, unsigned int idx) const;

    Vec thresholdParams;
};


struct Leontief : VecToScalar {
    // Perfect compliments
    Leontief(const Vec& productivities);
    virtual double f(const Vec& quantities) const;
    virtual double df(const Vec& quantities, unsigned int idx) const;

    Vec productivities;
};


struct CES : VecToScalar {
    // Constant elasticity of substitution
    // elast = 1 -> CobbDouglas
    // elast = infty -> Linear
    // elast = 0 -> Leontief
    CES(double tfp, const Vec& shareParams, double elasticityOfSubstitution);
    virtual double f(const Vec& quantities) const;
    virtual double df(const Vec& quantities, unsigned int idx) const;

    double tfp;
    Vec shareParams;
    double substitutionParam;
    double get_inner_sum(const Vec& quantities) const;
};


struct ProfitFunc : VecToScalar {
    // Encapsulates another VecToScalar to return the profit for different levels of production
    ProfitFunc(double price, const Vec& factorPrices, std::shared_ptr<VecToScalar> prodFunc);
    virtual double f(const Vec& quantities) const;
    virtual double df(const Vec& quantities, unsigned int idx) const;

    double price;
    std::shared_ptr<VecToScalar> prodFunc;
    Linear costFunc;
};

#endif
