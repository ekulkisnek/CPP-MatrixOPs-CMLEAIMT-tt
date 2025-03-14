#include "matrix.hpp"
#include "optimizations.hpp"
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace ml {

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    size_t size = get_aligned_size();
    data_ = std::make_unique<float[]>(size);
    std::fill_n(data_.get(), size, 0.0f);
}

Matrix::Matrix(const Matrix& other) 
    : rows_(other.rows_), cols_(other.cols_) {
    size_t size = get_aligned_size();
    data_ = std::make_unique<float[]>(size);
    std::copy_n(other.data_.get(), size, data_.get());
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        size_t size = get_aligned_size();
        auto new_data = std::make_unique<float[]>(size);
        std::copy_n(other.data_.get(), size, new_data.get());
        data_ = std::move(new_data);
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

Matrix Matrix::operator+(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_);
    std::copy_n(data_.get(), rows_ * cols_, result.data_.get());
    result.add_optimized(other);
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    validate_dimensions(other);
    Matrix result(rows_, cols_);
    std::copy_n(data_.get(), rows_ * cols_, result.data_.get());
    result.subtract_optimized(other);
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }
    Matrix result(rows_, other.cols_);
    block_multiply(result.data_.get(), data_.get(), other.data_.get(), 
                  rows_, other.cols_, cols_);
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    validate_dimensions(other);
    add_optimized(other);
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    validate_dimensions(other);
    subtract_optimized(other);
    return *this;
}

void Matrix::add_optimized(const Matrix& other) {
    validate_dimensions(other);
    simd_add(data_.get(), other.data_.get(), rows_ * cols_);
}

void Matrix::subtract_optimized(const Matrix& other) {
    validate_dimensions(other);
    simd_subtract(data_.get(), other.data_.get(), rows_ * cols_);
}

void Matrix::multiply_optimized(const Matrix& other) {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }
    Matrix result(rows_, other.cols_);
    block_multiply(result.data_.get(), data_.get(), other.data_.get(), 
                  rows_, other.cols_, cols_);
    data_ = std::move(result.data_);
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