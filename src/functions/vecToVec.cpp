#include "vecToVec.h"


unsigned int VecToVec::get_numInputs() const {
    return numInputs;
}

unsigned int VecToVec::get_numOutputs() const {
    return numOutputs;
}


VecToVecFromVecToScalar::VecToVecFromVecToScalar(std::shared_ptr<VecToScalar> vecToScalar) : vecToScalar(vecToScalar), VecToVec(vecToScalar->get_numInputs(), 1) {}

Vec VecToVecFromVecToScalar::f(const Vec& quantities) const {
    return Eigen::Array<double, 1, 1>(vecToScalar->f(quantities));
}

double VecToVecFromVecToScalar::df(const Vec& quantities, unsigned int i, unsigned int j) const {
    // i is irrelevant (it should always be zero)
    return df(quantities, j);
}

double VecToVecFromVecToScalar::df(const Vec& quantities, unsigned int j) const {
    return vecToScalar->df(quantities, j);
}
