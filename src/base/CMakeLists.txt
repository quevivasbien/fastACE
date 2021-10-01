target_sources(lib PRIVATE base.h economy.cpp agent.cpp firm.cpp person.cpp offers.cpp)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
