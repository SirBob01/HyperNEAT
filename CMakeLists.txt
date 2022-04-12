cmake_minimum_required(VERSION 3.10)
project(.)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

# HyperNEAT library files
file(GLOB_RECURSE INCLUDE "./src/*.h")
file(GLOB_RECURSE SOURCES "./src/*.cpp")

# Make the Brainiac engine a library
add_library(hyperneat STATIC ${SOURCES} ${INCLUDE})
target_include_directories(hyperneat PUBLIC "./src")

# Demo source files
file(GLOB DEMOS "./demos/*.cpp")

foreach(demo_path ${DEMOS})
    get_filename_component(demo_exe ${demo_path} NAME_WE)
    add_executable(${demo_exe} ${demo_path})

    target_link_libraries(${demo_exe} hyperneat)
endforeach()