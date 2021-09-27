#include <iostream>
#include "functions/solve.h"

int main() {
    Eigen::VectorXd zInit(2);
    zInit << 1, -1;
    testHimmelblau(zInit);
    return 0;
}
