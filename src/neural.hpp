#pragma once
#include "matrix.hpp"
#include <functional>
#include <vector>

namespace ml {

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh
};

class Layer {
public:
    Layer(size_t input_size, size_t output_size, ActivationType activation);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& gradient);
    void update_weights(float learning_rate);

    const Matrix& get_weights() const { return weights_; }
    const Matrix& get_biases() const { return biases_; }

private:
    Matrix weights_;
    Matrix biases_;
    Matrix last_input_;
    Matrix last_output_;
    ActivationType activation_;
    
    std::function<float(float)> get_activation_function();
    std::function<float(float)> get_activation_derivative();
};

class NeuralNetwork {
public:
    NeuralNetwork();
    
    void add_layer(size_t input_size, size_t output_size, ActivationType activation);
    Matrix forward(const Matrix& input);
    void backward(const Matrix& expected, float learning_rate);
    
    std::vector<Layer>& get_layers() { return layers_; }

private:
    std::vector<Layer> layers_;
    Matrix last_input_;
};

} // namespace ml
