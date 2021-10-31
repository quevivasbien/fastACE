find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_sources(lib PRIVATE decisionNets.h decisionNets.cpp)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

