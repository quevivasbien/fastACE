target_sources(lib PRIVATE vecToScalar.h vecToScalar.cpp vecToVec.h vecToVec.cpp)

find_package(Eigen3 REQUIRED)
target_link_libraries(lib PUBLIC Eigen3::Eigen)

target_include_directories(lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
