add_executable(main main.cpp)

add_library(lib SHARED "")

add_subdirectory(base)
add_subdirectory(persons)
add_subdirectory(firms)
add_subdirectory(functions)
add_subdirectory(torch)
add_subdirectory(pybindings)

target_link_libraries(main PUBLIC lib)


find_package (Threads REQUIRED)
target_link_libraries(lib PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
)


find_package(Eigen3 REQUIRED)
target_link_libraries(lib PUBLIC Eigen3::Eigen)

find_package(Torch REQUIRED)
target_link_libraries(lib PUBLIC "${TORCH_LIBRARIES}")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pyfastACE pybindings/pyfastACE.cpp)
target_compile_options(pyfastACE PRIVATE -fPIC)
target_link_libraries(pyfastACE PRIVATE lib)
