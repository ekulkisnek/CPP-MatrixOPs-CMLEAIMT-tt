
#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>

namespace ml {

// Matrix class: Implements a 2D matrix with SIMD-optimized operations
// Uses aligned memory allocation for AVX operations and implements
// basic matrix arithmetic with performance optimizations
class Matrix {
public:
    // Constructor: Creates a matrix with given dimensions
    // Memory is aligned for AVX operations (32-byte boundary)
    Matrix(size_t rows, size_t cols);
    
    // Copy constructor: Creates a deep copy of another matrix
    // Maintains memory alignment of the original
    Matrix(const Matrix& other);
    
    // Assignment operator: Performs deep copy while preserving alignment
    Matrix& operator=(const Matrix& other);
    
    // Destructor: Automatically handles aligned memory deallocation
    ~Matrix();

    // Arithmetic operations
    // Each returns a new matrix containing the result
    Matrix operator+(const Matrix& other) const; // Element-wise addition
    Matrix operator-(const Matrix& other) const; // Element-wise subtraction
    Matrix operator*(const Matrix& other) const; // Matrix multiplication

    // In-place arithmetic operations
    // Modifies the current matrix directly
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);

    // Element access with bounds checking
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    
    // Fills entire matrix with a single value
    void fill(float value);

    // Accessors for matrix properties
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    float* data() { return data_.get(); } // Direct data access for SIMD ops
    const float* data() const { return data_.get(); }

    // SIMD-optimized operations using AVX instructions
    void add_optimized(const Matrix& other);      // AVX-based addition
    void subtract_optimized(const Matrix& other); // AVX-based subtraction
    void multiply_optimized(const Matrix& other); // AVX-based multiplication

private:
    size_t rows_;    // Number of matrix rows
    size_t cols_;    // Number of matrix columns
    std::unique_ptr<float[]> data_; // Aligned memory for matrix elements
    
    // Utility functions
    void validate_dimensions(const Matrix& other) const; // Checks matrix compatibility
    size_t get_aligned_size() const; // Calculates size with padding for alignment
};

} // namespace ml
