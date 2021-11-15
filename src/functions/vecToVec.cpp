#include <assert.h>
#include "vecToVec.h"


// VToVFromVToS::VToVFromVToS(
//     std::shared_ptr<VecToScalar> vecToScalar,
//     unsigned int numOutputs,
//     unsigned int outputIndex
// ) : vecToScalar(vecToScalar), VecToVec(vecToScalar->numInputs, numOutputs), outputIndex(outputIndex) {}
//
// VToVFromVToS::VToVFromVToS(
//     std::shared_ptr<VecToScalar> vecToScalar
// ) : VToVFromVToS(vecToScalar, 1, 0) {}
//
// Vec VToVFromVToS::f(const Vec& quantities) const {
//     Vec out = Eigen::ArrayXd::Zero(numOutputs);
//     out(outputIndex) = vecToScalar->f(quantities);
//     return out;
// }
//
// double VToVFromVToS::df(const Vec& quantities, unsigned int i, unsigned int j) const {
//     if (i == outputIndex) {
//         return vecToScalar->df(quantities, j);
//     }
//     else {
//         return 0.0;
//     }
// }


SumOfVecToVec::SumOfVecToVec(
    std::vector<std::shared_ptr<VecToVec>> innerFunctions
) : innerFunctions(innerFunctions),
    numInnerFunctions(innerFunctions.size()),
    VecToVec(innerFunctions[0]->numInputs, innerFunctions[0]->numOutputs)
{
    for (unsigned int i = 1; i < numInnerFunctions; i++) {
        assert((innerFunctions[i]->numInputs == numInputs)
                && (innerFunctions[i]->numOutputs == numOutputs));
    }
}

Vec SumOfVecToVec::f(const Vec& quantities) const {
    Vec out = innerFunctions[0]->f(quantities);
    for (unsigned int i = 1; i < numInnerFunctions; i++) {
        out += innerFunctions[i]->f(quantities);
    }
    return out;
}

double SumOfVecToVec::df(const Vec& quantities, unsigned int i, unsigned int j) const {
    double out = innerFunctions[i]->df(quantities, i, j);
    for (unsigned int k = 1; k < numInnerFunctions; k++) {
        out += innerFunctions[k]->df(quantities, i, j);
    }
    return out;
}


std::shared_ptr<SumOfVecToVec> create_CES_VecToVec(
    std::vector<double> tfps,
    std::vector<Eigen::ArrayXd> shareParams,
    std::vector<double> elasticitiesOfSubstitution
) {
    unsigned int numGoods = tfps.size();
    assert((numGoods == shareParams.size()) && (numGoods == elasticitiesOfSubstitution.size()));

    std::vector<std::shared_ptr<VecToVec>> innerFunctions(numGoods);
    for (unsigned int i = 0; i < numGoods; i++) {
        innerFunctions[i] = std::make_shared<VToVFromVToS<CES>>(
            std::make_shared<CES>(
                tfps[i], shareParams[i], elasticitiesOfSubstitution[i]
            ),
            numGoods,
            i
        );
    }
    return std::make_shared<SumOfVecToVec>(innerFunctions);
}
