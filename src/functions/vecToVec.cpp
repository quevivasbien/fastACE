#include "vecToVec.h"


unsigned int VecToVec::get_numInputs() const {
    return numInputs;
}

unsigned int VecToVec::get_numOutputs() const {
    return numOutputs;
}


VToVFromVToS::VToVFromVToS(
    std::shared_ptr<VecToScalar> vecToScalar,
    unsigned int numOutputs,
    unsigned int outputIndex
) : vecToScalar(vecToScalar), VecToVec(vecToScalar->get_numInputs(), numOutputs), outputIndex(outputIndex) {}

VToVFromVToS::VToVFromVToS(
    std::shared_ptr<VecToScalar> vecToScalar
) : VToVFromVToS(vecToScalar, 1, 0) {}

Vec VToVFromVToS::f(const Vec& quantities) const {
    Vec out = Eigen::ArrayXd::Zero(numOutputs);
    out(outputIndex) = vecToScalar->f(quantities);
    return out;
}

double VToVFromVToS::df(const Vec& quantities, unsigned int i, unsigned int j) const {
    if (i == outputIndex) {
        return vecToScalar->df(quantities, j);
    }
    else {
        return 0.0;
    }
}
