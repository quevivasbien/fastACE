#ifndef VECTOVEC_H
#define VECTOVEC_H

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "vecToScalar.h"

using Vec = Eigen::ArrayXd;

struct VecToVec {
    virtual ~VecToVec() {}
    VecToVec(unsigned int numInputs, unsigned int numOutputs) : numInputs(numInputs), numOutputs(numOutputs) {}
    virtual Vec f(const Vec& quantities) const = 0;
    // df returns derivative of ith output w.r.t. jth input variable
    virtual double df(const Vec& quantities, unsigned int i, unsigned int j) const = 0;

    unsigned int numInputs;
    unsigned int numOutputs;
};


struct VToVFromVToS : VecToVec {
    // encloses a VecToScalar object but maintains functionality as a vec to vec
    // only returns a positive value in one of its output indices
    VToVFromVToS(
        std::shared_ptr<VecToScalar> vecToScalar,
        unsigned int numOutputs,
        unsigned int outputIndex
    );
    VToVFromVToS(
        std::shared_ptr<VecToScalar> vecToScalar
        // implicitly assumes numOutputs = 1 and outputIndex = 0
    );
    Vec f(const Vec& quantities) const override;
    double df(const Vec& quantities, unsigned int i, unsigned int j) const override;

    std::shared_ptr<VecToScalar> vecToScalar;
    unsigned int outputIndex;
};


struct SumOfVecToVec : VecToVec {
    // contains a list of VecToVecs with identical input and output dimensions
    // output will be sum of outputs for VecToVecs in the list
    SumOfVecToVec(std::vector<std::shared_ptr<VecToVec>> innerFunctions);
    Vec f(const Vec& quantities) const override;
    double df(const Vec& quantities, unsigned int i, unsigned int j) const override;

    std::vector<std::shared_ptr<VecToVec>> innerFunctions;
    unsigned int numInnerFunctions;
};

#endif
