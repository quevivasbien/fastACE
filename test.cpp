#include <iostream>
#include "economy.h"

int main() {
    Economy economy;
    economy.addPerson();
    economy.addFirm(economy.persons[0]);
    return 0;
}
