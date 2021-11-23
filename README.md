# fastACE

This is a library for agent-based computational economic modelling using deep reinforcement learning.

## Dependencies

* CMake for compilation
* [Eigen](https://eigen.tuxfamily.org/) for matrix and vector operations
* [LibTorch](https://pytorch.org/cppdocs/installing.html) for deep learning

On most Linux distributions, you can install CMake and Eigen via the command line, e.g.,
```bash
sudo apt install cmake libeigen3-dev
```

You can install LibTorch from https://pytorch.org/get-started/locally/. I recommend downloading the Libtorch version for CPU only (this package currently doesn't support training on a GPU); just download, extract the archive, and link to that archive when building the project.

## Configuring the project

Once you have the necessary libraries, navigate to the `build` dir, then run in the terminal
```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/torch/ ..
```
from the `build` directory, where `/path/to/torch` is the path to the folder where you extracted the LibTorch library.

### Notes

Note that if you have installed PyTorch for Python, you can use that instead of the standalone LibTorch package. Just run
```bash
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)' ..`
```
from the `build` directory. You may need to install some packages for NVIDIA's cuDNN in order for this option to work, depending on the version of PyTorch you have installed.

If Eigen is not in a place where CMake can find it automatically, you may need to provide its install directory as part of the `CMAKE_PREFIX_PATH` as well.

## Compiling the project

Once you've configured things, just run
```cmake --build . --config Release```
to compile the source code. Alternatively, just execute the `build.sh` script in the `build` directory.

This will generate a `main` executable in the `bin` directory. To change what the `main` executable does, you can edit `main.cpp` in the `src` directory.

## How to use

Forthcoming...


## Ifopt (optional)

If you want to program agents that use constrained optimization, you can use the code in `src/functions/solve.h`, which require `ifopt`. This is very much an optional feature, as none of the scripts included in the package currently make use of it by default.

See [the GitHub page](https://github.com/ethz-adrl/ifopt) for instructions on how to install `ifopt`. Note that installation is easy on Linux but quite difficult on Windows.

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
