# About this guide

This is not intended to be an exhaustive source of documentation for this project's code; rather, it's meant to provide a basic overview for where things are and what they do. You may still need to refer to the header files in the source code if you want details, at least for now.

This guide focuses on the C++ backend of this project. If you want information about the Python interface, you can find that in the [usage guide](./usage.md).

We'll start this guide by taking a brief look at the project's folder structure, then move on to the code's primitive class types, then some more advanced classes built on those types, then the machine learning functionalities that are currently implemented.

# Project structure

* bin: compiled libraries will be generated in this folder
* build: cmake build files will be generated in this folder
* docs: contains Markdown documentation for this project
* models: serialized Libtorch models will be saved in this folder
* py: contains Python scripts, including Python bindings for the C++ that makes up most of this project
* src: contains the C++ code for this project
    * base: code needed for minimal implementation of an agent-based economic simulation
    * firms: implementation of more advanced Firm classes
    * persons: implementation of more advanced Person classes
    * functions: containers for helpful scalar and vector functions
    * neural: code for simulations using reinforcement learning

# Primitive classes

The header file `src/base/base.h` defines the key classes that the library builds upon. These are the `Economy` and `Agent` classes. In short, an `Economy` is a container for a set of `Agents`. `Agents` represent economic actors -- persons or firms -- and make decisions while interacting with the other members of their parent `Economy`.

## The `Agent` class

A member of the `Agent` class represents an actor who can make decisions within an economy. The `Agent` class is pure virtual, meaning that you can't actually construct an `Agent`; rather it exists as an abstract type that defines the sorts of operations that actors in an economy can do -- these operations include things like creating offers to sell goods, searching through existing offers to buy goods, and providing information about internal state to other `Agent`s. You (the user) will almost never call any of an `Agent`s methods directly; rather, you will tell an `Economy` to manage the `Agent`s that you create, as discussed in the section on the `Economy` class.

All agents have a `money` attribute, which records their current wealth, as well as an `inventory` attribute, which is an Eigen array indicating the amount of each good in the economy that the agent currently holds.

The most important method implemented by `Agent` is probably `Agent::time_step()`, which tells the `Agent` to interact with the other members of its `Economy` and make decisions based on its current state.

There is also an `Agent::print_summary()` method, which displays some basic information about an `Agent`'s current state.

The `Person` and `Firm` classes inherit from `Agent` and therefore have access to all its methods.

## The `Person` class

A `Person` is an `Agent` that can buy and sell goods, supply labor to `Firm`s, and consume goods. When `Person::time_step()` is called, it does the following:

1. Check any offers that this `Person` has created to see if anyone has responded
2. Search for jobs (search through job offers from `Firm`s)
3. Buy goods (search through goods offers from other `Person`s and `Firms`s)
4. Consume goods

A `Person` cannot do anything without being assigned to an `Economy`. Therefore, you must provide a managing `Economy` when constructing a `Person`. To make this explicit, the `Person` class exposes a public `Person::init(...)` method that takes a pointer to an `Economy` and returns a shared pointer to a new `Person` instance that is managed by the provided `Economy`. Supposing we have an instance of `Economy` called `myEconomy`, we can create a new `Person`, managed by `myEconomy`, with `Person::init(&myEconomy)`.

The default implementation of `Person` doesn't really do anything interesting -- you can create instances of `Person` and call all of their methods, but nothing will happen. To define interesting behavior for the people in a simulation, you need to either overwrite the implementation of `Person`'s methods (i.e. by editing `src/base/person.cpp`) or, better, create a new class that inherits from `Person` and overrides its virtual methods. The `UtilMaxer` class, which is a handy version of a `Person` with a built-in utility function, does precisely this. The `UtilMaxer` class is discussed more in detail later in this guide.

## The `Firm` class

A `Firm` is an `Agent` that can buy and sell goods, hire labor from `Person`s, and use goods and labor to produce new goods. When `Firm::time_step()` is called, it does the following:

1. Check any offers (including job offers) that this `Firm` has created to see if anyone has responded
2. Buy goods (search through good offers from other `Person`s and `Firm`s)
3. Produce goods
4. Sell goods (create good offers)
5. Pay dividends
6. Search for laborers (create job offers)

Like with the `Person` class, you should use `Firm::init(...)` to create a new `Firm`.

A note on `Firm` ownership: `Firm`s can be assigned a list of other `Agent`s as its "owners." Although the current implementation doesn't do anything with firm ownership, you could potentially write code to make a `Firm`'s owners influence its decisions or receive dividends from its profits.

Like the `Person` class, the default implementation of `Firm` doesn't do anything interesting, so you should create a new class, inheriting from `Firm` to implement any desired behavior for the firms in a simulation. The `ProfitMaxer` class, which inherits from `Firm`, is a handy implemention of a firm with a built-in production function. The `ProfitMaxer` class is discussed more in detail later in this guide.

## Offers

Actors in the simulation buy and sell goods and labor by sharing offers, using a `BaseOffer` type, which has a subclass `Offer` for goods offers and `JobOffer` for job offers. If an `Agent` wants to sell some goods, it creates an instance of `Offer`, which it shares with its `Economy`. Other `Agent`s can then see that `Offer` and request the offerer for it. Similarly, `Firm`s wanting to hire laborers can create an instance of `JobOffer` to share with their `Economy` for `Person`s to view and request.

## The `Economy` class

To construct an economy, you need only supply a vector of good names. These goods will be the items traded, produced, and consumed within the economy. Below is an example of constructing an economy of rice and beans:

```c++
Economy riceAndBeansEconomy({"rice", "beans"});
```

An `Economy` won't do anything without any agents to manage. When we create `Person`s and `Firm`s via their `init()` methods, with a pointer to an `Economy` as the first argument, they will automatically be added to the list of `Agent`s managed by that `Economy`; for example, the following code creates a new `Person` and `Firm` that are managed by `riceAndBeansEconomy`:

```c++
Person::init(&riceAndBeansEconomy);
Firm::init(&riceAndBeansEconomy);
```

Any agents managed by an `Economy` will not go out of scope until the `Economy` goes out of scope.

An `Economy` also manages lists of `Offer` and `JobOffer` instances that are created by its managed `Agent`s. The `Agent`s will request to look at those lists, called the `market` and `jobMarket`, when they want to buy goods or find a job, respectively.

To make some action happen, we can call the `Economy::time_step()` method. This will tell each of the agents that the economy controls to call their respective `Agent::time_step()` methods and perform whatever actions that involves. An `Economy` is initialized with a time state of 0, which is incremented every time `Economy::time_step()` is called. To help make concurrency safe, `Economy::time_step()` will halt and return `false` if it detects that one of its managed agents is still working on something from a previous time step when it tries to step.

```c++
riceAndBeansEconomy.time_step();
// Loops through the `Person`s and `Firm`s managed by `riceAndBeansEconomy` and tells each of the to take a step in time
// Also clears out old offers in the goods market and job market
```

To check on the state of an `Economy`, we can call `Economy::print_summary()`, which will print something like the following:

```c++
riceAndBeansEconomy.print_summary();
// [Prints in console:]
// ----------
// Memory ID: 0x7fffa31c8f40 (Economy)
// ----------
// Time: 1

// Offers:
// Offerer: 0x5637dd08a950 ~ amt left: 2 ~ amt taken: 0
//  price: 1.99 ~ quantitities 1 1

// Job Offers:
// Offerer: 0x5637dd08a950 ~ amt left: 1 ~ amt taken: 0
//  wage: 5 ~ labor 0.5
```


# Derived classes

`UtilMaxer` is a child class of `Person` that comes equipped with a utility function. `ProfitMaxer` is a child class of `Firm` that comes equipped with a production function.

## `UtilMaxer`

`UtilMaxer` is defined in `src/persons/utilMaxer.h`. You can initialize an instance of `UtilMaxer` by calling its `init()` method, which takes arguments matching one of its constructors (note that its constructors are protected, so you can't call them directly; you must call `UtilMaxer::init()`, which in turn calls the corresponding constructor).

The `UtilMaxer` class adds two main features to the vanilla `Person` implementation: a utility function, and a decision maker helper class.

### The utility function

The `UtilMaxer`'s utility function is an instance of `VecToScalar`, which is discussed later in this guide. Essentially, `VecToScalar` is a wrapper for a function that takes multiple inputs and returns a scalar value. In the context of a utility function for a person, the inputs are a person's leisure for the current period (equal to 1 minus the agent's labor supplied), and the goods currently in the person's inventory. The utility function is stored as an attribute, `utilFunc` of the `UtilMaxer`. There is also a method, `UtilMaxer::u` provided for convenience, which gets the utility for given amounts of supplied labor and consumption.

Suppose that `utilMaxer` is a shared pointer to an instance of `UtilMaxer`. The following code snippet shows three equivalent ways of getting the utility that this person would receive from consuming its entire inventory.

```c++
// We can explicitly fetch the utilFunc member and call the function it wraps
double util1 = utilMaxer->get_utilFunc()->f(
    1.0 - utilMaxer->get_labor(),  // leisure = 1 - labor
    utilMaxer->get_inventory()
);
// A cleaner way is to call UtilMaxer::u
double util2 = utilMaxer->u(
    utilMaxer->get_labor(),
    utilMaxer->get_inventory()
);
// If we don't provide a value for labor, it's assumed we want utilMaxer->get_labor()
double util3 = utilMaxer->u(utilMaxer->get_inventory());

assert(util1 == util2 == util3);  // they're all equal
```

### The decision maker

`UtilMaxer` has an attribute, `decisionMaker`, which is an instance of the `PersonDecisionMaker` class. When a `UtilMaxer` needs to make a decision, it asks its decision maker what to do. As an example of how this works, the following is the implementation of `UtilMaxer::buy_goods()`:

```c++
void UtilMaxer::buy_goods() {
    auto orders = decisionMaker->choose_goods();
    for (auto order : orders) {
        for (unsigned int i = 0; i < order.amount; i++) {
            respond_to_offer(order.offer);
        }
    }
}
```

The `UtilMaxer` asks its decision maker which offers it should request, then requests those offers.

The point of this setup is to make it possible to get lots of different behaviors from a `UtilMaxer` without having to override its methods. Instead, to get the behavior you want, just create a new class inheriting from `PersonDecisionMaker` and implement its methods to do what you want. Note that `PersonDecisionMaker` itself is pure virtual, so you need to define a child class to use as a decision maker for a `UtilMaxer`.

## `ProfitMaxer`

`ProfitMaxer` class is analagous to `UtilMaxer`, but for `Firm`s.

`ProfitMaxer` is defined in `src/firms/profitMaxer.h`. You can initialize an instance of `ProfitMaxer` by calling its `init()` method, which takes arguments matching one of its constructors (note that its constructors are protected, so you can't call them directly; you must call `ProfitMaxer::init()`, which in turn calls the corresponding constructor).

The `ProfitMaxer` class adds two main features to the vanilla `Firm` implementation: a production function, and a decision maker helper class.

### The production function

The `ProfitMaxer`'s utility function is an instance of `VecToVec`, which is discussed later in this guide. Essentially, `VecToVec` is a wrapper for a function that takes multiple inputs and returns multiple outputs (in the form of an Eigen array). In the context of a production function for a firm, the inputs are the firm's labor hired for the current period, and the goods currently in the firm's inventory (the factors of production). The production function is stored as an attribute, `prodFunc` of the `ProfitMaxer`. There is also a method, `ProfitMaxer::f` provided for convenience, which gets the goods produced from given inputs.

Suppose that `profitMaxer` is a shared pointer to an instance of `ProfitMaxer`. The following code snippet shows two equivalent ways of getting the outputs that this firm would receive from using its entire inventory for production.

```c++
// We can explicitly fetch the prodFunc member and call the function it wraps
Eigen::ArrayXd goods1 = profitMaxer->get_prodFunc()->f(
    profitMaxer->get_labor(),
    profitMaxer->get_inventory()
);
// A cleaner way is to call profitMaxer::f
Eigen::ArrayXd goods2 = profitMaxer->f(
    profitMaxer->get_labor(),
    profitMaxer->get_inventory()
);

assert(goods1 == goods2);  // they're equal
```

### The decision maker

`ProfitMaxer` has an attribute, `decisionMaker`, which is an instance of the `FirmDecisionMaker` class. When a `ProfitMaxer` needs to make a decision, it asks its decision maker what to do. As an example of how this works, the following is a simplified version of the implementation of `ProfitMaxer::produce()`:

```c++
void ProfitMaxer::produce() {
    Eigen::ArrayXd inputs = decisionMaker->choose_production_inputs();
    inventory += f(laborHired, inputs) - inputs;
}
```

The `ProfitMaxer` asks its decision maker which goods in its inventory it should use for production, then produces based on that, updating its inventory to reflect that.

The point of this setup is to make it possible to get lots of different behaviors from a `ProfitMaxer` without having to override its methods. Instead, to get the behavior you want, just create a new class inheriting from `FirmDecisionMaker` and implement its methods to do what you want. Note that `FirmDecisionMaker` itself is pure virtual, so you need to define a child class to use as a decision maker for a `ProfitMaxer`.

It's worth noting that, although the name of the class is `ProfitMaxer`, you can implement `FirmDecisionMaker`s that will make a `ProfitMaxer` pursue an arbitrary objective, which may or may not be profit maximization.


# Function wrappers

The header files `src/functions/vecToScalar.h` and `src/functions/vecToVec.h` implement classes that wrap economically relevant functions with an interface for managing those functions' parameters. These are functions in the typical, mathematical, sense.

## `VecToScalar`

The `VecToScalar` class is a wrapper for functions taking an Eigen array as an input and returning a scalar (`double` type) value. The `VecToScalar::f` method is used to call the wrapped function. You can also use `VecToScalar::df` to call the function's derivatives.

`VecToScalar` is itself pure virtual; you can only instantiate instances of its child classes. Pre-implemented child classes include, among others, `CobbDouglas`, `Leontief`, `Linear`, and `CES`. Each child class has a distinct set of parameters that influence its behavior. The `CES` class in particular is used extensively within the machine learning-related code in the `src/neural` directory.

## `VecToVec`

The `VecToVec` class is a wrapper for functions taking an Eigen array as an input and returning an Eigen array as an output. Like with `VecToScalar`, you can use `VecToVec:f` to call the wrapped function and `VecToVec:df` to call its derivatives.

Again like `VecToScalar`, `VecToVec` is pure virtual -- it is intended only as a template for child classes. However, there aren't many child classes of `VecToVec` implemented by default; there is only `VToVFromVToS`, which wraps a `VecToScalar` instance, allowing it to produce an array output, and `SumOfVecToVec`, which wraps multiple `VecToScalar` instances, combining their outputs into an array output.


# Reinforcement learning

The `src/neural` directory contains code for implementing reinforcement learning in the context of an agent-based economic simulation. This is really the most exciting contribution of this project!

This code is based on the Libtorch library, which is the C++ frontend for PyTorch. If you're familiar with PyTorch, a lot of the neural network code here should look familiar, since Libtorch tries to maintain a style as similar as possible to its Python sibling.

## Basic structure

[in progress]


# Notes on code idioms

[in progress]