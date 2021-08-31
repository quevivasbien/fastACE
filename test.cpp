#include <iostream>
#include <vector>

class Object {
    int func() {
        return 0;
    }
};

struct Structure {
    Object* objPtr;
    int id = 1;
};

class Child : public Object {
public:
    Structure newFunc() {
        Structure str {this};
        return str;
    }
};

int main() {
    Child child;
    Structure str = child.newFunc();
    std::cout << &child << ' '<< str.objPtr << std::endl;
    return 0;
}
