cmake_minimum_required(VERSION 3.16)
project(mlcpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add source files
set(SOURCES
    src/matrix.cpp
    src/neural.cpp
    src/optimizations.cpp
    src/bindings.cpp
)

# Create the extension module
pybind11_add_module(mlcpp ${SOURCES})

# Enable AVX instructions
target_compile_options(mlcpp PRIVATE -mavx2 -mfma)

# Include directories
target_include_directories(mlcpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
