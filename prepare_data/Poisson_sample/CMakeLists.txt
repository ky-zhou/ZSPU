#cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS ON) #...without compiler extensions like gnu++11

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

include_directories(./vcglib-1.0.1)
include_directories(./vcglib-1.0.1/eigenlib)
add_executable(PdSampling Source.cpp)

#include_directories(./vcg)
#include_directories(./vcg/eigenlib)
#add_executable(PdSampling_nofix Source_nofix.cpp)

