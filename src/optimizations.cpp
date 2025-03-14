#include "optimizations.hpp"
#include <immintrin.h>
#include <algorithm>

namespace ml {

void simd_add(float* a, const float* b, size_t size) {
    size_t i = 0;

    // Process 8 elements at a time using AVX
    for (; i + 7 < size; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        __m256 result = _mm256_add_ps(va, vb);
        _mm256_store_ps(&a[i], result);
    }

    // Process remaining elements
    for (; i < size; ++i) {
        a[i] += b[i];
    }
}

void simd_subtract(float* a, const float* b, size_t size) {
    size_t i = 0;

    // Process 8 elements at a time using AVX
    for (; i + 7 < size; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        __m256 result = _mm256_sub_ps(va, vb);
        _mm256_store_ps(&a[i], result);
    }

    // Process remaining elements
    for (; i < size; ++i) {
        a[i] -= b[i];
    }
}

void block_multiply(float* result, const float* a, const float* b, size_t m, size_t n, size_t k) {
    constexpr size_t BLOCK_SIZE = 32;

    // Initialize result matrix to zero
    std::fill_n(result, m * n, 0.0f);

    for (size_t i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        size_t imax = std::min(i0 + BLOCK_SIZE, m);

        for (size_t j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            size_t jmax = std::min(j0 + BLOCK_SIZE, n);

            for (size_t k0 = 0; k0 < k; k0 += BLOCK_SIZE) {
                size_t kmax = std::min(k0 + BLOCK_SIZE, k);

                for (size_t i = i0; i < imax; ++i) {
                    for (size_t j = j0; j < jmax; ++j) {
                        float sum = 0.0f;
                        for (size_t k1 = k0; k1 < kmax; ++k1) {
                            sum += a[i * k + k1] * b[k1 * n + j];
                        }
                        result[i * n + j] += sum;
                    }
                }
            }
        }
    }
}

} // namespace ml