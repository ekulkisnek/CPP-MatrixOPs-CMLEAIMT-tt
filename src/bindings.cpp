#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix.hpp"
#include "neural.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mlcpp, m) {
    py::class_<ml::Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def("at", (float& (ml::Matrix::*)(size_t, size_t)) &ml::Matrix::at)
        .def("fill", &ml::Matrix::fill)
        .def("rows", &ml::Matrix::rows)
        .def("cols", &ml::Matrix::cols)
        .def("__add__", &ml::Matrix::operator+)
        .def("__sub__", &ml::Matrix::operator-)
        .def("__mul__", &ml::Matrix::operator*);

    py::enum_<ml::ActivationType>(m, "ActivationType")
        .value("ReLU", ml::ActivationType::ReLU)
        .value("Sigmoid", ml::ActivationType::Sigmoid)
        .value("Tanh", ml::ActivationType::Tanh);

    py::class_<ml::NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<>())
        .def("add_layer", &ml::NeuralNetwork::add_layer)
        .def("forward", &ml::NeuralNetwork::forward);
}
