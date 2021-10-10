#ifndef VECTOVEC_H
#define VECTOVEC_H

#include <memory>
#include <Eigen/Dense>
#include "vecToScalar.h"

using Vec = Eigen::ArrayXd;

class VecToVec {
public:
    virtual ~VecToVec() {}
    VecToVec(unsigned int numInputs, unsigned int numOutputs) : numInputs(numInputs), numOutputs(numOutputs) {}
    virtual Vec f(const Vec& quantities) const = 0;
    // df returns derivative of ith output w.r.t. jth input variable
    virtual double df(const Vec& quantities, unsigned int i, unsigned int j) const = 0;
    unsigned int get_numInputs() const;
    unsigned int get_numOutputs() const;
protected:
    unsigned int numInputs;
    unsigned int numOutputs;
};


class VToVFromVToS : public VecToVec {
    // encloses a VecToScalar object but maintains functionality as a vec to vec
    // only returns a positive value in one of its output indices
public:
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
protected:
    std::shared_ptr<VecToScalar> vecToScalar;
    unsigned int outputIndex;
};

#endif
