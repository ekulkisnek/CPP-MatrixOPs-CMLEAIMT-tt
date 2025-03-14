#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>

namespace ml {

class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    ~Matrix();

    // Basic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);

    // Element access
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    void fill(float value);

    // Properties
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }

    // SIMD operations
    void add_optimized(const Matrix& other);
    void subtract_optimized(const Matrix& other);
    void multiply_optimized(const Matrix& other);

private:
    size_t rows_;
    size_t cols_;
    std::unique_ptr<float[]> data_;
    
    void validate_dimensions(const Matrix& other) const;
    size_t get_aligned_size() const;
};

} // namespace ml
