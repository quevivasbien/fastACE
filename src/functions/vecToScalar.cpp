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


unsigned int VecToScalar::get_numInputs() const {
    return numInputs;
}

void VecToScalar::check_no_length_change(const Vec& candidate) const {
    assert(candidate.size() == numInputs);
}




Linear::Linear(const Vec& productivities) : VecToScalar(productivities.size()), productivities(productivities) {}

double Linear::f(const Vec& quantities) const {
    return (productivities * quantities).sum();
}

double Linear::df(const Vec& quantities, unsigned int idx) const {
    return productivities(idx);
}

void Linear::set_productivities(const Vec& newProductivities) {
    check_no_length_change(newProductivities);
    productivities = newProductivities;
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

void CobbDouglas::set_tfp(double new_tfp) {
    tfp = new_tfp;
}

void CobbDouglas::set_elasticities(const Vec& newElasticities) {
    check_no_length_change(newElasticities);
    elasticities = newElasticities;
}


CobbDouglasCRS::CobbDouglasCRS(double tfp, const Vec& elasticities) : CobbDouglas(tfp, elasticities) {
    this->elasticities /= elasticities.sum();
}

void CobbDouglasCRS::set_elasticities(const Vec& newElasticities) {
    check_no_length_change(newElasticities);
    elasticities = newElasticities / newElasticities.sum();
}




StoneGeary::StoneGeary(
    double tfp, const Vec& elasticities, const Vec& thresholdParams
) : CobbDouglas(tfp, elasticities), thresholdParams(thresholdParams) {
    assert(thresholdParams.size() == numInputs);
}

void StoneGeary::set_thresholdParams(const Vec& newThresholdParams) {
    check_no_length_change(newThresholdParams);
    thresholdParams = newThresholdParams;
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

void Leontief::set_productivities(const Vec& newProductivities) {
    check_no_length_change(newProductivities);
    productivities = newProductivities;
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


void CES::set_shareParams(const Vec& newShareParams) {
    check_no_length_change(newShareParams);
    shareParams = newShareParams;
}

void CES::set_substitutionParam(double newElasticityOfSubstitution) {
    substitutionParam = 1 / (1 - newElasticityOfSubstitution);
}




ProfitFunc::ProfitFunc(
    double price, const Vec& factorPrices, std::shared_ptr<VecToScalar> prodFunc
) : VecToScalar(factorPrices.size()), price(price), prodFunc(prodFunc), costFunc(Linear(factorPrices)) {
    //costFunc = Linear(factorPrices);
    assert(numInputs == this->prodFunc->get_numInputs());
}

double ProfitFunc::f(const Vec& quantities) const {
    return price * prodFunc->f(quantities) - costFunc.f(quantities);
}

double ProfitFunc::df(const Vec& quantities, unsigned int idx) const {
    return price * prodFunc->df(quantities, idx) - costFunc.df(quantities, idx);
}

void ProfitFunc::set_price(double newPrice) {
    price = newPrice;
}

void ProfitFunc::set_factorPrices(const Vec& newFactorPrices) {
    check_no_length_change(newFactorPrices);
    costFunc.set_productivities(newFactorPrices);
}

void ProfitFunc::set_prodFunc(std::shared_ptr<VecToScalar> newProdFunc) {
    assert(newProdFunc->get_numInputs() == numInputs);
    prodFunc = newProdFunc;
}
