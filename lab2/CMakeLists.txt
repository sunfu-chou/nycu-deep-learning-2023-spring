cmake_minimum_required(VERSION 3.0.0)
project(lab2 VERSION 0.1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin) 

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

file(GLOB_RECURSE SRCS "src/*.cpp")

enable_testing()

foreach(src IN LISTS SRCS)
    get_filename_component(target_name ${src} NAME_WE)
    add_executable("${target_name}" "${src}")
    endforeach()


# add_custom_target(run
#     COMMAND 2048
#     WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
# )
