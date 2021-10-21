# fastACE

This is a library for agent-based computational economic modelling.

## Dependencies

You will need CMake for compilation and [Eigen](https://eigen.tuxfamily.org/) for matrix and vector operations.

### Ifopt (optional)

If you want to use some of the constrained optimization functionalities, you will need `ifopt`. See [the GitHub page](https://github.com/ethz-adrl/ifopt) for instructions on how to install. Note that installation is easy on Linux but quite difficult on Windows.

On Debian/Ubuntu, you should just be able to run the following code to get things set up, assuming you've already installed CMake and Eigen:
```bash
# install ipopt
sudo apt-get install coinor-libipopt-dev
# clone the ifopt repo and compile
git clone https://github.com/ethz-adrl/ifopt.git && cd ifopt
mkdir build && cd build
cmake ..
make
sudo make install
```

Then navigate to `src/functions` and rename `CMakeLists.withifopt.txt` to `CMakeLists.txt` to tell CMake that the constrained optimization code should be compiled.

## Compilation

With all the dependencies installed, just run the following commands in the terminal to compile the library and create a `main` executable in the `bin` directory.
```bash
cd build
cmake ..
make
```
