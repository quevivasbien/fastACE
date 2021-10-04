#include "base.h"


int main() {
    std::vector<std::string> goods = {"wheat", "milk"};
    Economy economy(goods);
    auto person = economy.add_person();
    economy.add_firm(person);
    economy.time_step();
    return 0;
}
