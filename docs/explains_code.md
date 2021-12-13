# Primitive classes

The header file `src/base/base.h` defines the key classes that the library builds upon. These are the `Economy` and `Agent` classes. In short, an `Economy` is a container for a set of `Agents`. `Agents` represent economic actors -- persons or firms -- and make decisions while interacting with the other members of their parent `Economy`.

## The `Agent` class

A member of the `Agent` class represents an actor who can make decisions within an economy. The `Agent` class is pure virtual, meaning that you can't actually construct an `Agent`; rather it exists as an abstract type that defines the sorts of operations that actors in an economy can do -- these operations include things like creating offers to sell goods, searching through existing offers to buy goods, and providing information about internal state to other `Agent`s. You (the user) will almost never call any of an `Agent`s methods directly; rather, you will tell an `Economy` to manage the `Agent`s that you create, as discussed in the section on the `Economy` class. The most important method implemented by `Agent` is probably `Agent::time_step()`, which tells the `Agent` to interact with the other members of its `Economy` and make decisions based on its current state. The `Person` and `Firm` classes inherit from `Agent` and therefore have access to all its methods.

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

An `Economy` won't do anything without any agents to manage. When we create `Person`s and `Firm`s via their `init()` methods, with a pointer to an `Economy` as the first argument, they will automatically be added to the list of `Agent`s managed by that `Economy`; for example, the following code creates a new `Person` and `Firm` that are managed by `riceAndBeansEconomy`:c

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

[in progress]