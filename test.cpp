#include <iostream>
#include <memory>
#include "economy.h"

const int time_steps = 10;

int main() {
    Economy economy;
    economy.add_person();
    economy.persons[0]->add_money(10);
    economy.add_firm(economy.persons[0]);
    for (int i = 0; i < time_steps; i++) {
        std::cout << i << std::endl;
        economy.time_step();
        std::cout << economy.persons[0]->get_money() << std::endl;
        std::cout << economy.firms[0]->get_money() << std::endl;
    }
    return 0;
}
