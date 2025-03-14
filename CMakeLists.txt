
cmake_minimum_required(VERSION 3.10)
project(mlcpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(mlcpp 
    src/bindings.cpp
    src/matrix.cpp 
    src/neural.cpp
    src/optimizations.cpp
)

target_include_directories(mlcpp PRIVATE src)
target_compile_options(mlcpp PRIVATE -O3 -march=native)
