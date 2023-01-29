cmake_minimum_required(VERSION 3.24)
project(to_hzd)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(biped)

find_package(ifopt)

add_executable(to_hzd main.cpp)
target_link_libraries(to_hzd biped ifopt::ifopt_ipopt)
