#include "../src/matrix.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_matrix_creation() {
    ml::Matrix m(3, 4);
    assert(m.rows() == 3);
    assert(m.cols() == 4);
    
    // Test initialization
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            assert(m.at(i, j) == 0.0f);
        }
    }
}

void test_matrix_operations() {
    ml::Matrix a(2, 2);
    ml::Matrix b(2, 2);
    
    // Fill matrices
    a.at(0, 0) = 1.0f; a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f; a.at(1, 1) = 4.0f;
    
    b.at(0, 0) = 5.0f; b.at(0, 1) = 6.0f;
    b.at(1, 0) = 7.0f; b.at(1, 1) = 8.0f;
    
    // Test addition
    ml::Matrix c = a + b;
    assert(std::abs(c.at(0, 0) - 6.0f) < 1e-6);
    assert(std::abs(c.at(0, 1) - 8.0f) < 1e-6);
    assert(std::abs(c.at(1, 0) - 10.0f) < 1e-6);
    assert(std::abs(c.at(1, 1) - 12.0f) < 1e-6);
    
    // Test multiplication
    ml::Matrix d = a * b;
    assert(std::abs(d.at(0, 0) - 19.0f) < 1e-6);
    assert(std::abs(d.at(0, 1) - 22.0f) < 1e-6);
    assert(std::abs(d.at(1, 0) - 43.0f) < 1e-6);
    assert(std::abs(d.at(1, 1) - 50.0f) < 1e-6);
}

int main() {
    test_matrix_creation();
    test_matrix_operations();
    std::cout << "All matrix tests passed!" << std::endl;
    return 0;
}
