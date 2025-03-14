#pragma once
#include <cstddef>

namespace ml {

// SIMD-optimized operations
void simd_add(float* a, const float* b, size_t size);
void simd_subtract(float* a, const float* b, size_t size);
void simd_multiply(float* result, const float* a, const float* b, size_t m, size_t n, size_t k);

// Cache-friendly operations
void block_multiply(float* result, const float* a, const float* b, size_t m, size_t n, size_t k);

} // namespace ml
