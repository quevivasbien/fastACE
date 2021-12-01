target_sources(lib PRIVATE neuralConstants.h decisionNets.h decisionNets.cpp neuralEconomy.h neuralEconomy.cpp decisionNetHandler.cpp neuralPersonDecisionMaker.cpp neuralFirmDecisionMaker.cpp advantageActorCritic.h advantageActorCritic.cpp neuralScenarios.h)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
