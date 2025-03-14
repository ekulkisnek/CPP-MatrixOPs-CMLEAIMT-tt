#include "neural.hpp"
#include <cmath>
#include <random>

namespace ml {

Layer::Layer(size_t input_size, size_t output_size, ActivationType activation)
    : weights_(input_size, output_size)
    , biases_(1, output_size)
    , activation_(activation) {
    
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (input_size + output_size));
    std::uniform_real_distribution<float> dis(-limit, limit);
    
    for (size_t i = 0; i < input_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            weights_.at(i, j) = dis(gen);
        }
    }
    
    for (size_t i = 0; i < output_size; ++i) {
        biases_.at(0, i) = 0.0f;
    }
}

std::function<float(float)> Layer::get_activation_function() {
    switch (activation_) {
        case ActivationType::ReLU:
            return [](float x) { return x > 0 ? x : 0; };
        case ActivationType::Sigmoid:
            return [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        case ActivationType::Tanh:
            return [](float x) { return std::tanh(x); };
        default:
            throw std::runtime_error("Unknown activation function");
    }
}

std::function<float(float)> Layer::get_activation_derivative() {
    switch (activation_) {
        case ActivationType::ReLU:
            return [](float x) { return x > 0 ? 1.0f : 0.0f; };
        case ActivationType::Sigmoid:
            return [](float x) {
                float s = 1.0f / (1.0f + std::exp(-x));
                return s * (1.0f - s);
            };
        case ActivationType::Tanh:
            return [](float x) {
                float t = std::tanh(x);
                return 1.0f - t * t;
            };
        default:
            throw std::runtime_error("Unknown activation function");
    }
}

Matrix Layer::forward(const Matrix& input) {
    last_input_ = input;
    Matrix output = input * weights_;
    
    // Add biases
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output.at(i, j) += biases_.at(0, j);
        }
    }
    
    // Apply activation function
    auto activation = get_activation_function();
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output.at(i, j) = activation(output.at(i, j));
        }
    }
    
    last_output_ = output;
    return output;
}

NeuralNetwork::NeuralNetwork() = default;

void NeuralNetwork::add_layer(size_t input_size, size_t output_size, ActivationType activation) {
    layers_.emplace_back(input_size, output_size, activation);
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix current = input;
    for (auto& layer : layers_) {
        current = layer.forward(current);
    }
    return current;
}

} // namespace ml
