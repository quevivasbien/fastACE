add_executable(main main.cpp)

add_library(lib "")

add_subdirectory(base)
add_subdirectory(persons)
add_subdirectory(firms)
add_subdirectory(functions)
add_subdirectory(torch)
add_subdirectory(scenarios)


target_link_libraries(main PUBLIC lib)


find_package (Threads REQUIRED)
target_link_libraries(lib PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
)


find_package(Eigen3 REQUIRED)
target_link_libraries(lib PUBLIC Eigen3::Eigen)

find_package(Torch REQUIRED)
target_link_libraries(lib PUBLIC "${TORCH_LIBRARIES}")
