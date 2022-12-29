cmake_minimum_required(VERSION 3.0.2)
project(ORB)

SET(CMAKE_BUILD_TYPE "Debug")
add_compile_options(-std=c++11)
add_definitions(-std=c++11)

set(ALL_TARGET_LIBRARIES "")

include(cmake/eigen.cmake)
include(cmake/glog.cmake)
include(cmake/opencv.cmake)
include(cmake/YAML.cmake)
include(cmake/DBoW3.cmake)
include(cmake/global_defination.cmake)

include_directories(include)

add_library(${PROJECT_NAME} SHARED 
            src/ORBFeature.cpp
            src/parameter.cpp
            src/ORBextractor.cc
            src/trainingDictionary/trainingDic.cpp)      

target_link_libraries(${PROJECT_NAME} ${ALL_TARGET_LIBRARIES})

add_executable(hw8 src/apps/hw8.cpp)
target_link_libraries(hw8 ${PROJECT_NAME})

add_executable(hw91 src/apps/hw91.cpp)
target_link_libraries(hw91 ${PROJECT_NAME})

add_executable(hw92 src/apps/hw92.cpp)
target_link_libraries(hw92 ${PROJECT_NAME})
