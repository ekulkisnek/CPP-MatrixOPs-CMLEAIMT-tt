
#pragma once
#include "matrix.hpp"
#include <functional>
#include <vector>

namespace ml {

// Supported activation functions for neural network layers
enum class ActivationType {
    ReLU,    // Rectified Linear Unit: f(x) = max(0,x)
    Sigmoid, // Sigmoid function: f(x) = 1/(1+e^(-x))
    Tanh     // Hyperbolic tangent: f(x) = tanh(x)
};

// Layer class: Represents a single layer in the neural network
// Handles forward propagation, activation functions, and weight management
class Layer {
public:
    // Constructor: Initializes a layer with specified dimensions
    // Uses Xavier initialization for weights
    Layer(size_t input_size, size_t output_size, ActivationType activation);

    // Forward propagation: Computes layer output given input
    // Applies weights, biases, and activation function
    Matrix forward(const Matrix& input);
    
    // Backward propagation: Computes gradients for training
    Matrix backward(const Matrix& gradient);
    
    // Updates weights using computed gradients
    void update_weights(float learning_rate);

    // Accessors for layer parameters
    const Matrix& get_weights() const { return weights_; }
    const Matrix& get_biases() const { return biases_; }

private:
    Matrix weights_;      // Weight matrix (input_size x output_size)
    Matrix biases_;       // Bias vector (1 x output_size)
    Matrix last_input_;   // Cached input for backpropagation
    Matrix last_output_;  // Cached output for backpropagation
    ActivationType activation_; // Layer's activation function type

    // Helper functions for activation computations
    std::function<float(float)> get_activation_function() const;
    std::function<float(float)> get_activation_derivative() const;
};

// NeuralNetwork class: Manages multiple layers and network operations
class NeuralNetwork {
public:
    // Constructor: Creates an empty network
    NeuralNetwork();

    // Adds a new layer to the network
    void add_layer(size_t input_size, size_t output_size, ActivationType activation);
    
    // Forward propagation through all layers
    Matrix forward(const Matrix& input);
    
    // Backward propagation for training
    void backward(const Matrix& expected, float learning_rate);

    // Access to network layers
    const std::vector<Layer>& get_layers() const { return layers_; }

private:
    std::vector<Layer> layers_; // Sequential storage of network layers
    Matrix last_input_;         // Cached network input for training
};

} // namespace ml
