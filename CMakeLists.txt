cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(fastACE)

execute_process (
    COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE torchPath
)

execute_process (
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11Path
)

execute_process (
    COMMAND python3 -c "import distutils.sysconfig as ds; print(ds.get_python_inc())"
    OUTPUT_VARIABLE pythonPath
)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(SRC ${CMAKE_SOURCE_DIR}/src)

list(APPEND CMAKE_PREFIX_PATH ${torchPath} ${pybind11Path})
include_directories(${pythonPath})

add_subdirectory(${SRC} ${CMAKE_BINARY_DIR})

add_compile_options(-Wall)
add_compile_options(-fPIC)
