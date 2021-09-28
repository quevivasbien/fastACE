add_executable(main main.cpp)

add_library(lib "")

add_subdirectory(base)
add_subdirectory(persons)
add_subdirectory(functions)

target_link_libraries(main PUBLIC lib)
