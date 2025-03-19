
#pragma once
#include <cstddef>

namespace ml {

// SIMD-optimized operations using AVX instructions
// All functions assume properly aligned memory (32-byte boundary)

// Performs element-wise addition of two arrays using AVX
// a = a + b for all elements
void simd_add(float* a, const float* b, size_t size);

// Performs element-wise subtraction of two arrays using AVX
// a = a - b for all elements
void simd_subtract(float* a, const float* b, size_t size);

// Performs matrix multiplication using AVX instructions
// result = a * b where dimensions are (m,k) * (k,n) = (m,n)
void simd_multiply(float* result, const float* a, const float* b, 
                  size_t m, size_t n, size_t k);

// Cache-friendly block matrix multiplication
// Improves cache utilization by operating on small blocks
// result = a * b where dimensions are (m,k) * (k,n) = (m,n)
void block_multiply(float* result, const float* a, const float* b, 
                   size_t m, size_t n, size_t k);

} // namespace ml
