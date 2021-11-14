target_sources(lib PRIVATE decisionNets.h decisionNets.cpp decisionNetHandler.h decisionNetHandler.cpp neuralPersonDecisionMaker.h neuralPersonDecisionMaker.cpp neuralFirmDecisionMaker.h neuralFirmDecisionMaker.cpp neuralEconomy.h neuralEconomy.cpp)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
