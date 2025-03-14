
# Neural Network Matrix Operations Demo

A high-performance C++ implementation of matrix operations and neural networks with Python bindings using pybind11, featuring an interactive Streamlit visualization interface.

## Features

- **Optimized Matrix Operations**: 
  - AVX2-optimized matrix calculations
  - Cache-friendly block matrix multiplication
  - Basic operations (addition, subtraction, multiplication)
  - SIMD-accelerated computations

- **Neural Network Implementation**:
  - Modular layer architecture
  - Multiple activation functions (ReLU, Sigmoid, Tanh)
  - Xavier weight initialization
  - Forward propagation visualization

- **Interactive Visualization**:
  - Real-time matrix operation demonstrations
  - Neural network output surface plotting
  - Heat map visualizations of results
  - Dynamic input parameter adjustment

## Requirements

- Python 3.11+
- CMake
- C++ compiler with AVX2 support
- Required Python packages (automatically installed):
  - numpy
  - plotly
  - pybind11
  - streamlit
  - setuptools
  - wheel

## Quick Start

1. Click the "Run" button to build the C++ library and start the Streamlit app
2. Access the web interface through the generated URL
3. Experiment with different matrix operations and neural network configurations

## Usage

### Matrix Operations

The interface allows you to:
1. Specify matrix dimensions
2. Generate random matrices
3. Perform operations (Add, Subtract, Multiply)
4. Visualize results through heat maps

### Neural Network Demo

The demo showcases:
1. A simple 2-layer neural network (2→3→1 architecture)
2. ReLU and Sigmoid activation functions
3. 3D surface plot of the network's response

## Technical Details

### C++ Implementation

- Memory-aligned data structures for optimal performance
- SIMD vectorization using AVX2 instructions
- Block matrix multiplication for cache efficiency
- Exception-safe matrix operations

### Python Integration

- Seamless C++/Python binding using pybind11
- NumPy integration for efficient data transfer
- Streamlit-based interactive visualization

## Project Structure

```
├── src/                  # C++ source files
│   ├── matrix.hpp/cpp    # Matrix implementation
│   ├── neural.hpp/cpp    # Neural network implementation
│   ├── optimizations.cpp # SIMD optimizations
│   └── bindings.cpp      # Python bindings
├── tests/                # C++ unit tests
├── app.py               # Streamlit interface
├── setup.py            # Python package configuration
└── CMakeLists.txt     # CMake build configuration
```

## License

This project is available under the MIT License. See the LICENSE file for details.

## Contributing

Feel free to fork this template and make improvements. Pull requests are welcome!
