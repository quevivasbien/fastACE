target_sources(lib PRIVATE economy.h economy.cpp agent.cpp firm.cpp person.cpp)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
