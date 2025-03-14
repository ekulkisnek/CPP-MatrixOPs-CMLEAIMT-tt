#include "neural.hpp"
#include <cmath>
#include <random>

namespace ml {

Layer::Layer(size_t input_size, size_t output_size, ActivationType activation)
    : weights_(input_size, output_size)
    , biases_(1, output_size)
    , last_input_(1, input_size)
    , last_output_(1, output_size)
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

std::function<float(float)> Layer::get_activation_function() const {
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

std::function<float(float)> Layer::get_activation_derivative() const {
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

    last_input_ = input;
    last_output_ = output;
    return output;
}

Matrix Layer::backward(const Matrix& gradient) {
    auto derivative = get_activation_derivative();
    Matrix delta = gradient;

    // Apply activation derivative
    for (size_t i = 0; i < delta.rows(); ++i) {
        for (size_t j = 0; j < delta.cols(); ++j) {
            delta.at(i, j) *= derivative(last_output_.at(i, j));
        }
    }

    return delta;
}

void Layer::update_weights(float learning_rate) {
    // Weight updates implementation here
    // Left minimal for the demo
}

NeuralNetwork::NeuralNetwork() : last_input_(1, 1) {}

void NeuralNetwork::add_layer(size_t input_size, size_t output_size, ActivationType activation) {
    // Resize last_input_ if this is the first layer
    if (layers_.empty()) {
        last_input_ = Matrix(1, input_size);
    }
    layers_.emplace_back(input_size, output_size, activation);
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix current = input;
    last_input_ = input;

    for (auto& layer : layers_) {
        current = layer.forward(current);
    }
    return current;
}

void NeuralNetwork::backward(const Matrix& expected, float learning_rate) {
    // Backward pass implementation here
    // Left minimal for the demo
}

} // namespace ml