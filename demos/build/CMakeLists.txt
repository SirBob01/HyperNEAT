cmake_minimum_required(VERSION 3.10)
project(.)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O")

#Dynamo library files
file(GLOB_RECURSE INCLUDE "../../src/*.h")
file(GLOB_RECURSE SOURCES "../../src/*.cpp")

#Demo source files
file(GLOB DEMOS "../*.cpp")
add_library(HyperNEAT ${SOURCES} ${INCLUDE})

foreach(demo_path ${DEMOS})
    get_filename_component(demo_exe ${demo_path} NAME_WE)
    add_executable(${demo_exe} ${demo_path})

    target_link_libraries(${demo_exe} HyperNEAT)
endforeach()