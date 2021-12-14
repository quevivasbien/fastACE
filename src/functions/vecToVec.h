#ifndef VECTOVEC_H
#define VECTOVEC_H

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "vecToScalar.h"


class VecToVec {
public:
    virtual ~VecToVec() {}
    VecToVec(unsigned int numInputs, unsigned int numOutputs) : numInputs(numInputs), numOutputs(numOutputs) {}
    virtual Eigen::ArrayXd f(const Eigen::ArrayXd& quantities) const = 0;
    // df returns derivative of ith output w.r.t. jth input variable
    virtual double df(const Eigen::ArrayXd& quantities, unsigned int i, unsigned int j) const = 0;

    unsigned int numInputs;
    unsigned int numOutputs;
};


template <typename VToS>
class VToVFromVToS : public VecToVec {
    // encloses a VecToScalar object but maintains functionality as a vec to vec
    // only returns a positive value in one of its output indices
public:
    VToVFromVToS(
        std::shared_ptr<VToS> vecToScalar,
        unsigned int numOutputs,
        unsigned int outputIndex
    ) : vecToScalar(vecToScalar), VecToVec(vecToScalar->numInputs, numOutputs), outputIndex(outputIndex) {}

    VToVFromVToS(
        std::shared_ptr<VToS> vecToScalar
        // implicitly assumes numOutputs = 1 and outputIndex = 0
    ) : VToVFromVToS(vecToScalar, 1, 0) {}

    Eigen::ArrayXd f(const Eigen::ArrayXd& quantities) const override {
        Eigen::ArrayXd out = Eigen::ArrayXd::Zero(numOutputs);
        out(outputIndex) = vecToScalar->f(quantities);
        return out;
    }

    double df(const Eigen::ArrayXd& quantities, unsigned int i, unsigned int j) const override {
        if (i == outputIndex) {
            return vecToScalar->df(quantities, j);
        }
        else {
            return 0.0;
        }
    }

    std::shared_ptr<VToS> vecToScalar;
    unsigned int outputIndex;
};


class SumOfVecToVec : public VecToVec {
    // contains a list of VecToVecs with identical input and output dimensions
    // output will be sum of outputs for VecToVecs in the list
public:
    SumOfVecToVec(std::vector<std::shared_ptr<VecToVec>> innerFunctions);
    Eigen::ArrayXd f(const Eigen::ArrayXd& quantities) const override;
    double df(const Eigen::ArrayXd& quantities, unsigned int i, unsigned int j) const override;

    std::vector<std::shared_ptr<VecToVec>> innerFunctions;
    unsigned int numInnerFunctions;
};


// A convenient initializer for a SumOfVecToVecs made up of a CES production function for each output
// Need to provide vectors of length equal to number of goods in economy
std::shared_ptr<SumOfVecToVec> create_CES_VecToVec(
    std::vector<double> tfps,
    std::vector<Eigen::ArrayXd> shareParams,
    std::vector<double> elasticitiesOfSubstitution
);

#endif
