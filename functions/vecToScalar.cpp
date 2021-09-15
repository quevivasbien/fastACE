#include <vector>
#include <math.h>
#include <functional>
#include <stdexcept>
#include "vecToScalar.h"


double sum(const Vec& values) {
    double out = 0.0;
    for (const auto& value : values) {
        out += value;
    }
    return out;
}

double sum(const Vec& values, unsigned int length) {
    double out = 0.0;
    for (unsigned int i = 0; i < length; i++) {
        out += values[i];
    }
    return out;
}

double min(const Vec& values) {
    return min(values, values.size());
}

double min(const Vec& values, unsigned int length) {
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

double min(const Vec& values, unsigned int length, unsigned int* startIdx) {
    double out = values[*startIdx];
    if (length == *startIdx + 1) {
        return values[*startIdx];
    }
    for (unsigned int i = *startIdx + 1; i < length; i++) {
        if (values[i] > out) {
            out = values[i];
            *startIdx = i;
        }
    }
    return out;
}

void set_to_sum_one(Vec& vector) {
    return set_to_sum_one(vector, vector.size());
}

void set_to_sum_one(Vec& vector, unsigned int length) {
    double sumOf = sum(vector, length);
    for (unsigned int i = 0; i < length; i++) {
        vector[i] /= sumOf;
    }
}

Vec apply_to_each(Vec& vector, std::function<double(double)> func) {
    return apply_to_each(vector, func, vector.size());
}

Vec apply_to_each(Vec& vector, std::function<double(double)> func, unsigned int length) {
    Vec out(length);
    for (unsigned int i = 0; i < length; i++) {
        out[i] = func(vector[i]);
    }
    return out;
}

Vec multiply(const Vec& vector1, const Vec& vector2) {
    unsigned int length = vector1.size();
    if (length != vector2.size()) {
        throw std::runtime_error("From multiply: vector1 and vector2 must have same length");
    }
    return multiply(vector1, vector2, length);
}

Vec multiply(const Vec& vector1, const Vec& vector2, unsigned int length) {
    Vec out(length);
    for (unsigned int i = 0; i < length; i++) {
        out[i] = vector1[i] * vector2[i];
    }
    return out;
}


double VecToScalar::f(const Vec& quantities) {return 0.0;}

double VecToScalar::df(const Vec& quantities, unsigned int idx) {return 0.0;}


void VecToScalar::check_no_length_change(const Vec& candidate) {
    if (candidate.size() != numInputs) {
        throw std::runtime_error(
            "From VecToScalar: You cannot change the length of a parameter vector after initialization."
        );
    }
}


CobbDouglas::CobbDouglas(double tfp, Vec elasticities) : tfp(tfp), elasticities(elasticities) {
    numInputs = elasticities.size();
}

double CobbDouglas::f(const Vec& quantities) {
    double out = tfp;
    for (unsigned int i = 0; i < numInputs; i++) {
        out *= pow(quantities[i], elasticities[i]);
    }
    return out;
}

double CobbDouglas::df(const Vec& quantities, unsigned int idx) {
    double out = tfp;
    for (unsigned int i = 0; i < numInputs; i++) {
        if (i == idx) {
            out *= (elasticities[i] * pow(quantities[i], elasticities[i] - 1));
        }
        else {
            out *= pow(quantities[i], elasticities[i]);
        }
    }
    return out;
}

void CobbDouglas::set_tfp(double new_tfp) {
    tfp = new_tfp;
}

void CobbDouglas::set_elasticities(Vec newElasticities) {
    check_no_length_change(newElasticities);
    elasticities = newElasticities;
}


CobbDouglasCRS::CobbDouglasCRS(double tfp, Vec elasticities) : CobbDouglas(tfp, elasticities) {
    set_to_sum_one(elasticities);
}

void CobbDouglasCRS::set_elasticities(Vec newElasticities) {
    check_no_length_change(newElasticities);
    elasticities = newElasticities;
    set_to_sum_one(elasticities);
}


StoneGeary::StoneGeary(
    double tfp, Vec elasticities, Vec thresholdParams
) : CobbDouglas(tfp, elasticities), thresholdParams(thresholdParams) {
    if (thresholdParams.size() != numInputs) {
        throw std::runtime_error("From StoneGeary: elasticities and thresholdParams must have same size");
    }
}

void StoneGeary::set_thresholdParams(Vec newThresholdParams) {
    check_no_length_change(newThresholdParams);
    thresholdParams = newThresholdParams;
}


double StoneGeary::f(const Vec& quantities) {
    double out = tfp;
    for (unsigned int i = 0; i < numInputs; i++) {
        out *= pow(quantities[i] - thresholdParams[i], elasticities[i]);
    }
    return out;
}

double StoneGeary::df(const Vec& quantities, unsigned int idx) {
    double out = tfp;
    for (unsigned i = 0; i < numInputs; i++) {
        if (i == idx) {
            out *= (elasticities[i] * pow(quantities[i] - thresholdParams[i], elasticities[i] - 1));
        }
        else {
            out *= pow(quantities[i] - thresholdParams[i], elasticities[i]);
        }
    }
    return out;
}


Leontief::Leontief(Vec productivities) : productivities(productivities) {
    numInputs = productivities.size();
}

double Leontief::f(const Vec& quantities) {
    return min(multiply(quantities, productivities, numInputs), numInputs);
}

double Leontief::df(const Vec& quantities, unsigned int idx) {
    Vec values = multiply(quantities, productivities, numInputs);
    unsigned int minIdx = 0;
    double minVal = min(values, numInputs, &minIdx);
    if (minIdx != idx) {
        return 0.0;
    }
    else {
        minIdx++;
        if (min(values, numInputs, &minIdx) == minVal) {
            return 0.0;
        }
        else {
            return productivities[idx];
        }
    }
}

void Leontief::set_productivities(Vec newProductivities) {
    check_no_length_change(newProductivities);
    productivities = newProductivities;
}


CES::CES(
    double tfp, Vec shareParams, double elasticityOfSubstitution
) : tfp(tfp), shareParams(shareParams), substitutionParam(1 / (1-elasticityOfSubstitution)) {
    numInputs = shareParams.size();
}

double CES::get_inner_sum(const Vec& quantities) {
    double innerSum = 0.0;
    for (unsigned int i = 0; i < numInputs; i++) {
        innerSum += shareParams[i] * pow(quantities[i], substitutionParam);
    }
    return innerSum;
}

double CES::f(const Vec& quantities) {
    return tfp * pow(get_inner_sum(quantities), 1 / substitutionParam);
}

double CES::df(const Vec& quantities, unsigned int idx) {
    double innerSum = get_inner_sum(quantities);
    return tfp * pow(innerSum, 1 / substitutionParam - 1)
        * shareParams[idx] * pow(quantities[idx], substitutionParam - 1);
}


void CES::set_shareParams(Vec newShareParams) {
    check_no_length_change(newShareParams);
    shareParams = newShareParams;
}

void CES::set_substitutionParam(double newElasticityOfSubstitution) {
    substitutionParam = 1 / (1 - newElasticityOfSubstitution);
}
