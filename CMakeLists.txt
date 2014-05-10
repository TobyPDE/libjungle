cmake_minimum_required(VERSION 2.8)
project(DecisionJungle)
find_package( Boost 1.55.0 COMPONENTS filesystem system REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
add_executable(jungleTrain jungleTrain.cpp jungle.cpp )
add_executable(junglePredict junglePredict.cpp jungle.cpp )

target_link_libraries(jungleTrain 
    ${Boost_LIBRARIES}
)
target_link_libraries(junglePredict
    ${Boost_LIBRARIES}
)
ADD_DEFINITIONS(
    -std=c++11
)