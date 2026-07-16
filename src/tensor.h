#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

class Tensor {
public:
    float* data;   // device pointer to the actual values
    float* grad;   // device pointer to this tensor's gradient (same shape as data)
    int rows, cols;

    Tensor(int rows_, int cols_) : rows(rows_), cols(cols_) {
        cudaMalloc(&data, rows * cols * sizeof(float));
        cudaMalloc(&grad, rows * cols * sizeof(float));
        cudaMemset(grad, 0, rows * cols * sizeof(float)); // gradients start at zero
    }

    ~Tensor() {
        cudaFree(data);
        cudaFree(grad);
    }

    int size() const { return rows * cols; }
};

#include <functional>

inline std::vector<std::function<void()>> tape;

inline void backwardAll() {
    for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
        (*it)();
    }
    tape.clear();
}
