set ( lib_jungle_SOURCES jungleTrain.cpp jungle.cpp )
add_library ( lib_jungle ${lib_jungle_SOURCES} )
cmake_minimum_required(VERSION 2.8)
find_package( Boost 1.55.0 COMPONENTS system REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
configure_file (
  "${PROJECT_SOURCE_DIR}/src/config.h.in"
  "${PROJECT_SOURCE_DIR}/src/config.h"
)
target_link_libraries( lib_jungle ${Boost_LIBRARIES} )
add_definitions ( -std=c++0x -O3 )