#include <iostream>
#include "economy.h"
#include "vecToScalar.h"

int main() {
    Economy economy;
    std::cout << economy.market.size() << std::endl;
    Vec v(2);
    v << 0.5, 0.5;
    std::cout << v.transpose() << std::endl;
    return 0;
}
