
#pragma once
#include <cstddef>

namespace ml {

// All functions in this file implement SIMD (Single Instruction Multiple Data)
// optimizations using AVX (Advanced Vector Extensions) instructions
// Memory must be aligned to 32-byte boundaries for optimal performance

// Performs vectorized element-wise addition of two arrays
// a: Destination array (will be modified)
// b: Source array
// size: Number of elements
// Uses AVX __m256 to process 8 floats simultaneously
void simd_add(float* a, const float* b, size_t size);

// Performs vectorized element-wise subtraction of two arrays
// a: Destination array (will be modified)
// b: Source array to subtract
// size: Number of elements
// Uses AVX __m256 to process 8 floats simultaneously
void simd_subtract(float* a, const float* b, size_t size);

// Performs optimized matrix multiplication using AVX instructions
// result: Output matrix (m x n)
// a: First input matrix (m x k)
// b: Second input matrix (k x n)
// m, n, k: Matrix dimensions
// Uses AVX for vectorized multiply-add operations
void simd_multiply(float* result, const float* a, const float* b, 
                  size_t m, size_t n, size_t k);

// Implements cache-friendly block matrix multiplication
// Improves cache utilization by operating on small blocks that fit in L1/L2 cache
// result: Output matrix (m x n)
// a: First input matrix (m x k)
// b: Second input matrix (k x n)
// m, n, k: Matrix dimensions
void block_multiply(float* result, const float* a, const float* b, 
                   size_t m, size_t n, size_t k);

} // namespace ml
