target_sources(lib PRIVATE util.h base.h constants.h economy.cpp agent.cpp firm.cpp person.cpp offers.cpp scenario.h)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
