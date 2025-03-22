
#pragma once
#include "matrix.hpp"
#include <functional>
#include <vector>

namespace ml {

// Enumeration of supported activation functions for neural network layers
enum class ActivationType {
    ReLU,    // Rectified Linear Unit f(x) = max(0,x) - Good for hidden layers
    Sigmoid, // Logistic function f(x) = 1/(1+e^(-x)) - Good for binary classification
    Tanh     // Hyperbolic tangent f(x) = tanh(x) - Alternative to sigmoid
};

// Layer class: Represents a single neural network layer
// Handles forward propagation, activation, and weight management
class Layer {
public:
    // Initialize a new layer with specified dimensions and activation
    // input_size: Number of input neurons
    // output_size: Number of output neurons
    // activation: Type of activation function to use
    Layer(size_t input_size, size_t output_size, ActivationType activation);

    // Compute layer output for given input
    // input: Matrix of input values (batch_size x input_size)
    // Returns: Matrix of output values (batch_size x output_size)
    Matrix forward(const Matrix& input);
    
    // Compute gradients for backpropagation
    // gradient: Gradient from next layer
    // Returns: Gradient to pass to previous layer
    Matrix backward(const Matrix& gradient);
    
    // Update layer weights using computed gradients
    // learning_rate: Step size for gradient descent
    void update_weights(float learning_rate);

    // Accessor methods for layer parameters
    const Matrix& get_weights() const { return weights_; }  // Get weight matrix
    const Matrix& get_biases() const { return biases_; }   // Get bias vector

private:
    Matrix weights_;      // Weight matrix (input_size x output_size)
    Matrix biases_;       // Bias vector (1 x output_size)
    Matrix last_input_;   // Cache of last input for backprop
    Matrix last_output_;  // Cache of last output for backprop
    ActivationType activation_; // Type of activation function

    // Helper functions for activation computations
    // Returns function pointer to activation implementation
    std::function<float(float)> get_activation_function() const;
    // Returns function pointer to activation derivative
    std::function<float(float)> get_activation_derivative() const;
};

// NeuralNetwork class: Manages multiple layers and network operations
class NeuralNetwork {
public:
    // Create empty network ready for layer addition
    NeuralNetwork();

    // Add new layer to network
    // input_size: Number of inputs to layer
    // output_size: Number of outputs from layer
    // activation: Activation function type
    void add_layer(size_t input_size, size_t output_size, ActivationType activation);
    
    // Process input through all network layers
    // input: Input matrix (batch_size x input_size)
    // Returns: Output matrix (batch_size x output_size_of_last_layer)
    Matrix forward(const Matrix& input);
    
    // Train network using backpropagation
    // expected: Expected output values
    // learning_rate: Learning rate for weight updates
    void backward(const Matrix& expected, float learning_rate);

    // Get all network layers
    const std::vector<Layer>& get_layers() const { return layers_; }

private:
    std::vector<Layer> layers_; // Sequential storage of network layers
    Matrix last_input_;         // Cache of network input for training
};

} // namespace ml
