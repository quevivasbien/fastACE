# fastACE

This is a library for agent-based computational economic modelling.

## To install

You will need [`ifopt`](https://github.com/ethz-adrl/ifopt) to use the optimization functionalities. See the GitHub page for instructions on how to install. Note that installation is easy on Linux but very difficult on Windows.

`ifopt` itself uses [Eigen](https://eigen.tuxfamily.org/) for linear algebra and other math. You'll probably get that during the process of installing `ifopt`, but otherwise, you'll need to install it yourself.

To compile (assuming you have cMake):
```bash
cd build
cmake ..
make
```
