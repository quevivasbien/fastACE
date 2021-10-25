cmake_minimum_required(VERSION 3.13)

project(fastACE)

option(INCLUDE_IFOPT "Compile code that requires ifopt" ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(SRC ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(src ${CMAKE_BINARY_DIR})

add_compile_options(-Wall -pthread)
