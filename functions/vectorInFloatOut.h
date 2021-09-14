#ifndef HOMOTHETIC_H
#define HOMOTHETIC_H

#include <vector>
#include <functional>


double sum(const std::vector<double>& values);
double sum(const std::vector<double>& values, unsigned int length);

double min(const std::vector<double>& values);
double min(const std::vector<double>& values, unsigned int length);

void set_to_sum_one(std::vector<double>& vector);
void set_to_sum_one(std::vector<double>& vector, unsigned int length);

std::vector<double> apply_to_each(std::vector<double>& vector, std::function<double(double)> func);
std::vector<double> apply_to_each(std::vector<double>& vector, std::function<double(double)> func, unsigned int length);

std::vector<double> multiply(const std::vector<double>& vector1, const std::vector<double>& vector2);
std::vector<double> multiply(const std::vector<double>& vector1, const std::vector<double>& vector2, unsigned int length);


class VectorInFloatOut {
    // base class meant to store parameters for a real-valued function
public:
    double f(const std::vector<double>& quantities);
    virtual double f(const std::vector<double>& quantities, unsigned int inputLength);
protected:
    void check_no_length_change(const std::vector<double>& candidate);
    unsigned int numInputs;
};


class CobbDouglas : public VectorInFloatOut {
public:
    CobbDouglas(double tfp, std::vector<double> elasticities);
    virtual double f(const std::vector<double>& quantities, unsigned int inputLength);
    void set_tfp(double newTfp);
    virtual void set_elasticities(std::vector<double> newElasticities);
protected:
    double tfp;
    std::vector<double> elasticities;
};


class CobbDouglasCRS : public CobbDouglas {
public:
    CobbDouglasCRS(double tfp, std::vector<double> elasticities);
    void set_elasticities(std::vector<double> newElasticities);
};


class StoneGeary : public CobbDouglas {
public:
    StoneGeary(double tfp, std::vector<double> elasticities, std::vector<double> thresholdParams);
    virtual double f(const std::vector<double>& quantities, unsigned int inputLength);
    void set_thresholdParams(std::vector<double> newThresholdParams);
protected:
    std::vector<double> thresholdParams;
};


class Leontief : public VectorInFloatOut {
public:
    Leontief(std::vector<double> productivities);
    virtual double f(const std::vector<double>& quantities, unsigned int inputLength);
    void set_productivities(std::vector<double> newProductivities);
protected:
    std::vector<double> productivities;
};


class CES : public VectorInFloatOut {
public:
    CES(double tfp, std::vector<double> shareParams, double elasticityOfSubstitution);
    virtual double f(const std::vector<double>& quantities, unsigned int inputLength);
    void set_shareParams(std::vector<double> newShareParams);
    void set_substitutionParam(double newElasticityOfSubstitution);
protected:
    double tfp;
    std::vector<double> shareParams;
    double substitutionParam;
};

#endif
