
#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>

namespace ml {

// Matrix class: Core implementation of matrix operations optimized for SIMD/AVX2
// All memory is aligned to 32-byte boundaries for optimal AVX operations
class Matrix {
public:
    // Creates a new matrix with specified dimensions
    // Memory is allocated with 32-byte alignment for AVX2 SIMD operations
    // rows: Number of matrix rows
    // cols: Number of matrix columns
    Matrix(size_t rows, size_t cols);
    
    // Deep copy constructor - creates exact duplicate of source matrix
    // Maintains 32-byte memory alignment of the original
    // other: Source matrix to copy from
    Matrix(const Matrix& other);
    
    // Assignment operator - performs deep copy while keeping alignment
    // Returns reference to allow chained assignments (a = b = c)
    Matrix& operator=(const Matrix& other);
    
    // Cleanup memory - automatically handles aligned deallocation
    ~Matrix();

    // Mathematical Operations - all return new matrices
    // const qualifier ensures these don't modify the original matrices
    Matrix operator+(const Matrix& other) const; // Element-wise matrix addition
    Matrix operator-(const Matrix& other) const; // Element-wise matrix subtraction
    Matrix operator*(const Matrix& other) const; // Matrix multiplication

    // In-place operations - modify the current matrix
    // Returns reference to allow operation chaining (a += b += c)
    Matrix& operator+=(const Matrix& other); // Add other matrix to this one
    Matrix& operator-=(const Matrix& other); // Subtract other matrix from this one

    // Access individual elements with bounds checking
    // row: Zero-based row index
    // col: Zero-based column index
    // Throws std::out_of_range if indices are invalid
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    
    // Set all elements to specified value
    // value: The value to fill the matrix with
    void fill(float value);

    // Accessor methods for matrix properties
    size_t rows() const { return rows_; }  // Get number of rows
    size_t cols() const { return cols_; }  // Get number of columns
    float* data() { return data_.get(); }  // Get raw data pointer for SIMD ops
    const float* data() const { return data_.get(); }  // Const version

    // SIMD-optimized operations using AVX2
    void add_optimized(const Matrix& other);      // AVX2 vectorized addition
    void subtract_optimized(const Matrix& other); // AVX2 vectorized subtraction
    void multiply_optimized(const Matrix& other); // AVX2 vectorized multiplication

private:
    size_t rows_;    // Number of matrix rows
    size_t cols_;    // Number of matrix columns
    std::unique_ptr<float[]> data_; // Aligned memory buffer for matrix elements
    
    // Internal helper methods
    void validate_dimensions(const Matrix& other) const; // Check matrix compatibility
    size_t get_aligned_size() const; // Calculate size with padding for AVX2
};

} // namespace ml
