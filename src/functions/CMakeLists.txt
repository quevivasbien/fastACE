target_sources(lib PRIVATE vecToScalar.h vecToScalar.cpp vecToVec.h vecToVec.cpp)

find_package(Eigen3 REQUIRED)
if(INCLUDE_IFOPT)
    target_sources(lib PRIVATE solve.h solve.cpp)
    find_package(ifopt REQUIRED)
    target_link_libraries(lib PUBLIC ifopt::ifopt_ipopt Eigen3::Eigen)
else()
    target_link_libraries(lib PUBLIC Eigen3::Eigen)
endif(INCLUDE_IFOPT)
unset(INCLUDE_IFOPT CACHE)

target_include_directories(lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
