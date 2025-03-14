#include "../src/neural.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_layer_creation() {
    ml::Layer layer(4, 3, ml::ActivationType::ReLU);
    assert(layer.get_weights().rows() == 4);
    assert(layer.get_weights().cols() == 3);
    assert(layer.get_biases().rows() == 1);
    assert(layer.get_biases().cols() == 3);
}

void test_forward_propagation() {
    ml::NeuralNetwork nn;
    nn.add_layer(2, 3, ml::ActivationType::ReLU);
    nn.add_layer(3, 1, ml::ActivationType::Sigmoid);
    
    ml::Matrix input(1, 2);
    input.at(0, 0) = 1.0f;
    input.at(0, 1) = 2.0f;
    
    ml::Matrix output = nn.forward(input);
    assert(output.rows() == 1);
    assert(output.cols() == 1);
    assert(output.at(0, 0) >= 0.0f && output.at(0, 0) <= 1.0f);
}

int main() {
    test_layer_creation();
    test_forward_propagation();
    std::cout << "All neural network tests passed!" << std::endl;
    return 0;
}
