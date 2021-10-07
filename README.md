# fastACE

This is a library for agent-based computational economic modelling.

## To install

You will need [`ifopt`](https://github.com/ethz-adrl/ifopt) to use the optimization functionalities. See the GitHub page for instructions on how to install. Note that installation is easy on Linux but very difficult on Windows.

On Debian/Ubuntu, you should just be able to run the following code to get things set up:
```bash
# get some dependencies
sudo apt-get install cmake libeigen3-dev coinor-libipopt-dev
# clone the ifopt repo and compile
git clone https://github.com/ethz-adrl/ifopt.git && cd ifopt
mkdir build && cd build
cmake ..
make
sudo make install
```

To compile (assuming you have cMake):
```bash
cd build
cmake ..
make
```

