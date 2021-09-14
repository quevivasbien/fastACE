#include "../economy.h"

// a bit odd, this is meant to be a class that contains a function f,
// and _parameters_ for that function
// this doesn't contain anything right now; it's just a parent class
class Function {};


class OwnedFunction {
    // basically just a wrapper around Function that has a concept of ownership
    // e.g. a Person can have a utility function, which only he can edit
private:
    Agent* owner;


}
