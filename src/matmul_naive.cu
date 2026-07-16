#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

const int M = 2048;
const int K = 2048;
const int N = 2048;

__global__ void matmulNaiveKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k *N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);

    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand() % 10);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmulNaiveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Cuda error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost);

    std::vector<float> h_C_ref(M * N, 0.0f);
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[row * K + k] * h_B[k * N + col];
            }
            h_C_ref[row * N + col] = sum;
        }
    }

    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (std::fabs(h_C[i] - h_C_ref[i]) > 1e-2f) {
            std::cerr << "Mismatch at " << i << ": got " << h_C[i]
                      << ", expected " << h_C_ref[i] << std::endl;
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "PASS" : "FAIL") << ": matmul " << M << "x" << K
              << " * " << K << "x" << N << std::endl;
    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
