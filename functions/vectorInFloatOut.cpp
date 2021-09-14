#include <vector>
#include <math.h>
#include <functional>
#include <stdexcept>
#include "vectorInFloatOut.h"


double sum(const std::vector<double>& values) {
    double out = 0.0;
    for (const auto& value : values) {
        out += value;
    }
    return out;
}

double sum(const std::vector<double>& values, unsigned int length) {
    double out = 0.0;
    for (unsigned int i = 0; i < length; i++) {
        out += values[i];
    }
    return out;
}

double min(const std::vector<double>& values) {
    return min(values, values.size());
}

double min(const std::vector<double>& values, unsigned int length) {
    if (length == 1) {
        return values[0];
    }
    double out = values[0];
    for (unsigned int i = 1; i < length; i++) {
        if (values[i] < out) {
            out = values[i];
        }
    }
    return out;
}

void set_to_sum_one(std::vector<double>& vector) {
    return set_to_sum_one(vector, vector.size());
}

void set_to_sum_one(std::vector<double>& vector, unsigned int length) {
    double sumOf = sum(vector, length);
    for (unsigned int i = 0; i < length; i++) {
        vector[i] /= sumOf;
    }
}

std::vector<double> apply_to_each(std::vector<double>& vector, std::function<double(double)> func) {
    return apply_to_each(vector, func, vector.size());
}

std::vector<double> apply_to_each(std::vector<double>& vector, std::function<double(double)> func, unsigned int length) {
    std::vector<double> out(length);
    for (unsigned int i = 0; i < length; i++) {
        out[i] = func(vector[i]);
    }
    return out;
}

std::vector<double> multiply(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    unsigned int length = vector1.size();
    if (length != vector2.size()) {
        throw std::runtime_error("From multiply: vector1 and vector2 must have same length");
    }
    return multiply(vector1, vector2, length);
}

std::vector<double> multiply(const std::vector<double>& vector1, const std::vector<double>& vector2, unsigned int length) {
    std::vector<double> out(length);
    for (unsigned int i = 0; i < length; i++) {
        out[i] = vector1[i] * vector2[i];
    }
    return out;
}


double VectorInFloatOut::f(const std::vector<double>& values) {
    return f(values, values.size());
}

double VectorInFloatOut::f(const std::vector<double>& values, unsigned int inputLength) {
    // not implemented
    return 0.0;
}

void VectorInFloatOut::check_no_length_change(const std::vector<double>& candidate) {
    if (candidate.size() != numInputs) {
        throw std::runtime_error(
            "From VectorInFloatOut: You cannot change the length of a parameter vector after initialization."
        );
    }
}


CobbDouglas::CobbDouglas(double tfp, std::vector<double> elasticities) : tfp(tfp), elasticities(elasticities) {
    numInputs = elasticities.size();
}

double CobbDouglas::f(const std::vector<double>& quantities, unsigned int inputLength) {
    double out = tfp;
    for (unsigned int i = 0; i < inputLength; i++) {
        out *= pow(quantities[i], elasticities[i]);
    }
    return out;
}

void CobbDouglas::set_tfp(double new_tfp) {
    tfp = new_tfp;
}

void CobbDouglas::set_elasticities(std::vector<double> newElasticities) {
    check_no_length_change(newElasticities);
    elasticities = newElasticities;
}


CobbDouglasCRS::CobbDouglasCRS(double tfp, std::vector<double> elasticities) : CobbDouglas(tfp, elasticities) {
    set_to_sum_one(elasticities);
}

void CobbDouglasCRS::set_elasticities(std::vector<double> newElasticities) {
    check_no_length_change(newElasticities);
    elasticities = newElasticities;
    set_to_sum_one(elasticities);
}


StoneGeary::StoneGeary(
    double tfp, std::vector<double> elasticities, std::vector<double> thresholdParams
) : CobbDouglas(tfp, elasticities), thresholdParams(thresholdParams) {
    if (thresholdParams.size() != numInputs) {
        throw std::runtime_error("From StoneGeary: elasticities and thresholdParams must have same size");
    }
}

void StoneGeary::set_thresholdParams(std::vector<double> newThresholdParams) {
    check_no_length_change(newThresholdParams);
    thresholdParams = newThresholdParams;
}


double StoneGeary::f(const std::vector<double>& quantities, unsigned int inputLength) {
    double out = tfp;
    for (unsigned int i = 0; i < inputLength; i++) {
        out *= pow(quantities[i] - thresholdParams[i], elasticities[i]);
    }
    return out;
}


Leontief::Leontief(std::vector<double> productivities) : productivities(productivities) {
    numInputs = productivities.size();
}

double Leontief::f(const std::vector<double>& quantities, unsigned int inputLength) {
    return min(multiply(quantities, productivities));
}

void Leontief::set_productivities(std::vector<double> newProductivities) {
    check_no_length_change(newProductivities);
    productivities = newProductivities;
}


CES::CES(
    double tfp, std::vector<double> shareParams, double elasticityOfSubstitution
) : tfp(tfp), shareParams(shareParams), substitutionParam(1 / (1-elasticityOfSubstitution)) {
    numInputs = shareParams.size();
}

double CES::f(const std::vector<double>& quantities, unsigned int inputLength) {
    double innerSum = 0.0;
    for (unsigned int i = 0; i < inputLength; i++) {
        innerSum += shareParams[i] * pow(quantities[i], substitutionParam);
    }
    return tfp * pow(innerSum, 1 / substitutionParam);
}

void CES::set_shareParams(std::vector<double> newShareParams) {
    check_no_length_change(newShareParams);
    shareParams = newShareParams;
}

void CES::set_substitutionParam(double newElasticityOfSubstitution) {
    substitutionParam = 1 / (1 - newElasticityOfSubstitution);
}
