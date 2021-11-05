#include <math.h>
#include <assert.h>
#include "vecToScalar.h"


double min(const Vec& values, unsigned int length, unsigned int* startIdx) {
    double out = values(*startIdx);
    if (length == *startIdx + 1) {
        return values(*startIdx);
    }
    for (unsigned int i = *startIdx + 1; i < length; i++) {
        if (values(i) > out) {
            out = values(i);
            *startIdx = i;
        }
    }
    return out;
}

void VecToScalar::check_no_length_change(const Vec& candidate) const {
    assert(candidate.size() == numInputs);
}



Linear::Linear(unsigned int numInputs) : VecToScalar(numInputs), productivities(Eigen::ArrayXd::Constant(numInputs, 1.0)) {}
Linear::Linear(const Vec& productivities) : VecToScalar(productivities.size()), productivities(productivities) {}

double Linear::f(const Vec& quantities) const {
    return (productivities * quantities).sum();
}

double Linear::df(const Vec& quantities, unsigned int idx) const {
    return productivities(idx);
}


CobbDouglas::CobbDouglas(
    unsigned int numInputs
) : VecToScalar(numInputs), tfp(1.0), elasticities(Eigen::ArrayXd::Constant(numInputs, 1.0 / numInputs)) {}

CobbDouglas::CobbDouglas(double tfp, const Vec& elasticities) : VecToScalar(elasticities.size()), tfp(tfp), elasticities(elasticities) {}

double CobbDouglas::f(const Vec& quantities) const {
    return tfp * Eigen::pow(quantities, elasticities).prod();
}

double CobbDouglas::df(const Vec& quantities, unsigned int idx) const {
    return f(quantities) * elasticities(idx) / quantities(idx);
}


CobbDouglasCRS::CobbDouglasCRS(double tfp, const Vec& elasticities) : CobbDouglas(tfp, elasticities) {
    this->elasticities /= elasticities.sum();
}



StoneGeary::StoneGeary(
    double tfp, const Vec& elasticities, const Vec& thresholdParams
) : CobbDouglas(tfp, elasticities), thresholdParams(thresholdParams) {
    assert(thresholdParams.size() == numInputs);
}


double StoneGeary::f(const Vec& quantities) const {
    return tfp * Eigen::pow(quantities - thresholdParams, elasticities).prod();
}

double StoneGeary::df(const Vec& quantities, unsigned int idx) const {
    return f(quantities) * elasticities(idx) / (quantities(idx) - thresholdParams(idx));
}




Leontief::Leontief(const Vec& productivities) : VecToScalar(productivities.size()), productivities(productivities) {}

double Leontief::f(const Vec& quantities) const {
    return (quantities * productivities).minCoeff();
}

double Leontief::df(const Vec& quantities, unsigned int idx) const {
    const Vec& values = quantities * productivities;
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
            return productivities(idx);
        }
    }
}




CES::CES(
    double tfp, const Vec& shareParams, double elasticityOfSubstitution
) : VecToScalar(shareParams.size()), tfp(tfp), shareParams(shareParams), substitutionParam(1 / (1-elasticityOfSubstitution)) {}

double CES::get_inner_sum(const Vec& quantities) const {
    return (shareParams * Eigen::pow(quantities, substitutionParam)).sum();
}

double CES::f(const Vec& quantities) const {
    return tfp * pow(get_inner_sum(quantities), 1 / substitutionParam);
}

double CES::df(const Vec& quantities, unsigned int idx) const {
    double innerSum = get_inner_sum(quantities);
    return tfp * pow(innerSum, 1 / substitutionParam - 1)
        * shareParams(idx) * pow(quantities(idx), substitutionParam - 1);
}




ProfitFunc::ProfitFunc(
    double price, const Vec& factorPrices, std::shared_ptr<VecToScalar> prodFunc
) : VecToScalar(factorPrices.size()), price(price), prodFunc(prodFunc), costFunc(Linear(factorPrices)) {
    //costFunc = Linear(factorPrices);
    assert(numInputs == this->prodFunc->numInputs);
}

double ProfitFunc::f(const Vec& quantities) const {
    return price * prodFunc->f(quantities) - costFunc.f(quantities);
}

double ProfitFunc::df(const Vec& quantities, unsigned int idx) const {
    return price * prodFunc->df(quantities, idx) - costFunc.df(quantities, idx);
}
