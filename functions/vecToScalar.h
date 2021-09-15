#ifndef VECTOSCALAR_H
#define VECTOSCALAR_H

#include <vector>
#include <functional>

using Vec = std::vector<double>;

// returns sum of values
double sum(const Vec& values);
// same thing, with length of values pre-specified
double sum(const Vec& values, unsigned int length);

// returns minimum of values
double min(const Vec& values);
// '' with length of values pre-specified
double min(const Vec& values, unsigned int length);
// '' with starting index given as a pointer, starting index will be modified in place to be the index of the max value
double min(const Vec& values, unsigned int length, unsigned int* startIdx);

// normalizes a vector so that it sums to one
void set_to_sum_one(Vec& vector);
// '' length given...
void set_to_sum_one(Vec& vector, unsigned int length);

// applies a scalar function to each element of a vector
Vec apply_to_each(Vec& vector, std::function<double(double)> func);
Vec apply_to_each(Vec& vector, std::function<double(double)> func, unsigned int length);

// multiplies two equal-length vectors elementwise
Vec multiply(const Vec& vector1, const Vec& vector2);
Vec multiply(const Vec& vector1, const Vec& vector2, unsigned int length);


class VecToScalar {
    // base class meant to store parameters for a real-valued function & its derivatives
public:
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
protected:
    void check_no_length_change(const Vec& candidate);
    unsigned int numInputs;
};


class CobbDouglas : public VecToScalar {
public:
    CobbDouglas(double tfp, Vec elasticities);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_tfp(double newTfp);
    virtual void set_elasticities(Vec newElasticities);
protected:
    double tfp;
    Vec elasticities;
};


class CobbDouglasCRS : public CobbDouglas {
public:
    CobbDouglasCRS(double tfp, Vec elasticities);
    void set_elasticities(Vec newElasticities);
};


class StoneGeary : public CobbDouglas {
public:
    StoneGeary(double tfp, Vec elasticities, Vec thresholdParams);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_thresholdParams(Vec newThresholdParams);
protected:
    Vec thresholdParams;
};


class Leontief : public VecToScalar {
public:
    Leontief(Vec productivities);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_productivities(Vec newProductivities);
protected:
    Vec productivities;
};


class CES : public VecToScalar {
public:
    CES(double tfp, Vec shareParams, double elasticityOfSubstitution);
    virtual double f(const Vec& quantities);
    virtual double df(const Vec& quantities, unsigned int idx);
    void set_shareParams(Vec newShareParams);
    void set_substitutionParam(double newElasticityOfSubstitution);
protected:
    double tfp;
    Vec shareParams;
    double substitutionParam;
    double get_inner_sum(const Vec& quantities);
};

#endif
