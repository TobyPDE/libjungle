cmake_minimum_required(VERSION 2.8)
project(DecisionJungle)
set (DecisionJungle_VERSION_MAJOR 1)
set (DecisionJungle_VERSION_MINOR 0)
find_package( Boost 1.55.0 COMPONENTS system REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
add_executable(jungle cliInterface.cpp jungleTrain.cpp jungle.cpp )
configure_file (
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_SOURCE_DIR}/config.h"
)
target_link_libraries(jungle 
    ${Boost_LIBRARIES}
)
ADD_DEFINITIONS(
    -std=c++0x
    -O3
)