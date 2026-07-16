#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

const int M = 256;
const int K = 256;
const int N = 256;

__global__ void matmulForwardKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmulBackwardKernel(const float* dC, const float* B, float* dA, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M
    int k = blockIdx.x * blockDim.x + threadIdx.x; // 0..K

    if (row < M && k < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            sum += dC[row * N + n] * B[k * N + n];
        }
        dA[row * K + k] = sum;
    }
}

__global__ void matmulBackwardBKernel(const float* A, const float* dC, float* dB, int M, int K, int N) {
    int k = blockIdx.y * blockDim.y + threadIdx.y; // 0..K
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N

    if (k < K && col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) {
            sum += A[m * K + k] * dC[m * N + col];
        }
        dB[k * N + col] = sum;
    }
}

int main() {
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M *N);
    std::vector<float> h_dC(M * N), h_dA(M * K), h_dB(K * N);

    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < M * N; i++) h_dC[i] = static_cast<float>((rand() % 200) - 100) / 100.0f;

    float *d_A, *d_B, *d_C, *d_dC, *d_dA, *d_dB;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
    cudaMalloc(&d_dC, bytesC);
    cudaMalloc(&d_dA, bytesA);
    cudaMalloc(&d_dB, bytesB);

    cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dC, h_dC.data(), bytesC, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);

    dim3 gridC((N + 15) / 16, (M + 15) / 16);
    matmulForwardKernel<<<gridC, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    dim3 gridA((K + 15) / 16, (M + 15) / 16);
    matmulBackwardKernel<<<gridA, threadsPerBlock>>>(d_dC, d_B, d_dA, M, K, N);
    cudaDeviceSynchronize();

    dim3 gridB((N + 15) / 16, (K + 15) / 16);
    matmulBackwardBKernel<<<gridB, threadsPerBlock>>>(d_A, d_dC, d_dB, M, K, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Cuda error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dA.data(), d_dA, bytesA, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dB.data(), d_dB, bytesB, cudaMemcpyDeviceToHost);

    // CPU reference: dA = dC * B^T
    std::vector<float> h_dA_ref(M * K, 0.0f);
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                sum += h_dC[row * N + n] * h_B[k * N + n];
            }
            h_dA_ref[row * K + k] = sum;
        }
    }

    // CPU reference: dB = A^T * dC
    std::vector<float> h_dB_ref(K * N, 0.0f);
    for (int k = 0; k < K; k++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int m = 0; m < M; m++) {
                sum += h_A[m * K + k] * h_dC[m * N + col];
            }
            h_dB_ref[k * N + col] = sum;
        }
    }

    bool correct = true;
    for (int i = 0; i < M * K; i++) {
        if (std::fabs(h_dA[i] - h_dA_ref[i]) > 1e-1f) {
            std::cerr << "dA mistmatch at " << i << ": got " << h_dA[i] << ", expected " << h_dA_ref[i] << std::endl;
            correct = false;
            break;
        }
    }
    for (int i = 0; i < K * N; i++) {
        if (std::fabs(h_dB[i] - h_dB_ref[i]) > 1e-1f) {
            std::cerr << "dB mustmatch at " << i << ": got " << h_dB[i] << ", expected " << h_dB_ref[i] << std::endl;
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "PASS" : "FAIL") << ": matmul backward (dA, dB), " << M << "x" << K << " * " << K << "x" << N << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_dC);
    cudaFree(d_dA);
    cudaFree(d_dB);

    return 0;
}
