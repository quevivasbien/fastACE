cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(fastACE)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(SRC ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(${SRC} ${CMAKE_BINARY_DIR})
