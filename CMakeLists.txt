cmake_minimum_required(VERSION 3.13)

project(fastACE)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/obj)
set(SRC ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(src ${CMAKE_BINARY_DIR})
