cmake_minimum_required(VERSION 3.0.0)
project(lab2 VERSION 0.1.0)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin) 

enable_testing()

add_executable(2048 src/2048_sample.cpp)
