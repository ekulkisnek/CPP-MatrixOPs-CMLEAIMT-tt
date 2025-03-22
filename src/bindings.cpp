
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix.hpp"
#include "neural.hpp"

namespace py = pybind11;  // Alias for pybind11 namespace

// Define Python module 'mlcpp' and expose C++ classes/functions
PYBIND11_MODULE(mlcpp, m) {
    // Expose Matrix class to Python
    py::class_<ml::Matrix>(m, "Matrix")
        // Constructor with rows and columns
        .def(py::init<size_t, size_t>())
        // Element access method - returns reference to allow modification
        .def("at", (float& (ml::Matrix::*)(size_t, size_t)) &ml::Matrix::at)
        // Matrix filling method
        .def("fill", &ml::Matrix::fill)
        // Dimension accessors
        .def("rows", &ml::Matrix::rows)
        .def("cols", &ml::Matrix::cols)
        // Operator overloads for Python
        .def("__add__", &ml::Matrix::operator+)  // Matrix addition
        .def("__sub__", &ml::Matrix::operator-)  // Matrix subtraction
        .def("__mul__", &ml::Matrix::operator*); // Matrix multiplication

    // Expose ActivationType enum to Python
    py::enum_<ml::ActivationType>(m, "ActivationType")
        .value("ReLU", ml::ActivationType::ReLU)       // Rectified Linear Unit
        .value("Sigmoid", ml::ActivationType::Sigmoid) // Sigmoid activation
        .value("Tanh", ml::ActivationType::Tanh);     // Hyperbolic tangent

    // Expose NeuralNetwork class to Python
    py::class_<ml::NeuralNetwork>(m, "NeuralNetwork")
        // Default constructor
        .def(py::init<>())
        // Layer addition method
        .def("add_layer", &ml::NeuralNetwork::add_layer)
        // Forward propagation method
        .def("forward", &ml::NeuralNetwork::forward);
}
