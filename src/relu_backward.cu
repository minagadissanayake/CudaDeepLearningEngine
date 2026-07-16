#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

const int N = 1 << 20; // ~1 million elements, same scale as vector_add

__global__ void reluForwardKernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

__global__ void reluBackwardKernel(const float* x, const float* dL_dy, float* dL_dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dL_dx[i] = (x[i] > 0.0f) ? dL_dy[i] : 0.0f;
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    std::vector<float> h_x(N), h_y(N), h_dL_dy(N), h_dL_dx(N);

    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>((rand() % 2000) - 1000) / 100.0f; // range -10 to 10
        h_dL_dy[i] = static_cast<float>((rand() % 200) - 100) / 100.0f; // range -1 to 1
    }

    float *d_x, *d_y, *d_dL_dy, *d_dL_dx;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_dL_dy, bytes);
    cudaMalloc(&d_dL_dx, bytes);

    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dL_dy, h_dL_dy.data(), bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reluForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    reluBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_dL_dy, d_dL_dx, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Cuda error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dL_dx.data(), d_dL_dx, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected_y = (h_x[i] > 0.0f) ? h_x[i] : 0.0f;
        float expected_dL_dx = (h_x[i] > 0.0f) ? h_dL_dy[i] : 0.0f;

        if (std::fabs(h_y[i] - expected_y) > 1e-5f ||
            std::fabs(h_dL_dx[i] - expected_dL_dx) > 1e-5f) {
            std::cerr << "Mismatch at " << i << std::endl;
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "PASS" : "FAIL") << ": relu forward + backward, " << N << " elements." << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dL_dy);
    cudaFree(d_dL_dx);

    return 0;
}
