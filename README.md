# fastACE

This is a library for agent-based computational economic modelling.

## To install

Clone this repository and use `make` to compile all the code. You can edit the `makefile` to change the main script, output and included modules.

The only dependency is [Eigen](https://eigen.tuxfamily.org/) for linear algebra and other math. You will need to download it and include the path to the Eigen directory as a flag when compiling (i.e. `-I path/to/Eigen`). Note that there is no need to pre-compile the Eigen library since all of its modules are contained purely in headers.
