#include "base.h"
#include "utilMaxer.h"


int main() {
    std::vector<std::string> goods = {"wheat", "milk"};
    Economy economy(goods);
    auto person = create<UtilMaxer>(&economy);
    economy.add_firm(person);
    economy.time_step();
    person->print_summary();
    return 0;
}
