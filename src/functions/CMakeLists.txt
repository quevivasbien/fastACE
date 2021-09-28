target_sources(lib PRIVATE solve.h vecToScalar.h vecToScalar.cpp)

find_package(Eigen3 REQUIRED)
find_package(ifopt REQUIRED)
target_link_libraries(lib PUBLIC ifopt::ifopt_ipopt Eigen3::Eigen)

target_include_directories(lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
