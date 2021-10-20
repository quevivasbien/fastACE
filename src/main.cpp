#include "base.h"
#include "utilMaxer.h"
#include "profitMaxer.h"


int main() {
    std::vector<std::string> goods = {"wheat", "milk"};
    Economy economy(goods);
    auto person = create<UtilMaxer>(&economy);
    auto firm = ProfitMaxer::init(&economy, person, 0);
    economy.add_agent(person);
    economy.add_agent(firm);
    economy.time_step();
    person->print_summary();
    return 0;
}
