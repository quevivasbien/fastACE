#include <iostream>
#include <memory>
#include "economy.h"

const int timeSteps = 10;

int main() {
    Economy economy;
    economy.addPerson();
    economy.persons[0]->addMoney(10);
    economy.addFirm(economy.persons[0]);
    for (int i = 0; i < timeSteps; i++) {
        std::cout << i << std::endl;
        economy.timeStep();
        std::cout << economy.persons[0]->getMoney() << std::endl;
        std::cout << economy.firms[0]->getMoney() << std::endl;
    }
    return 0;
}
