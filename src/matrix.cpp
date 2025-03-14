#include "matrix.hpp"
#include "optimizations.hpp"
#include <algorithm>

namespace ml {

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    size_t size = get_aligned_size();
    data_ = std::make_unique<float[]>(size);
    std::fill_n(data_.get(), rows_ * cols_, 0.0f);
}

Matrix::Matrix(const Matrix& other) 
    : rows_(other.rows_), cols_(other.cols_) {
    size_t size = get_aligned_size();
    data_ = std::make_unique<float[]>(size);
    std::memcpy(data_.get(), other.data_.get(), rows_ * cols_ * sizeof(float));
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            size_t size = get_aligned_size();
            data_ = std::make_unique<float[]>(size);
        }
        std::memcpy(data_.get(), other.data_.get(), rows_ * cols_ * sizeof(float));
    }
    return *this;
}

Matrix::~Matrix() = default;

float& Matrix::at(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[row * cols_ + col];
}

const float& Matrix::at(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[row * cols_ + col];
}

void Matrix::fill(float value) {
    std::fill_n(data_.get(), rows_ * cols_, value);
}

Matrix Matrix::operator+(const Matrix& other) {
    validate_dimensions(other);
    Matrix result(rows_, cols_);
    result.add_optimized(other);
    return result;
}

Matrix Matrix::operator-(const Matrix& other) {
    validate_dimensions(other);
    Matrix result(rows_, cols_);
    result.subtract_optimized(other);
    return result;
}

Matrix Matrix::operator*(const Matrix& other) {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }
    Matrix result(rows_, other.cols_);
    result.multiply_optimized(other);
    return result;
}

void Matrix::validate_dimensions(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
}

size_t Matrix::get_aligned_size() const {
    // Align to 32-byte boundary for AVX
    return ((rows_ * cols_ * sizeof(float) + 31) / 32) * 32 / sizeof(float);
}

} // namespace ml
