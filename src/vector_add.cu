#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// ---------------------------------------------------------
// The kernel: this function runs on the GPU. __global__ means
// "callable from the CPU, executes on the device."
// Every GPU thread that launches runs THIS SAME CODE, but with
// a different threadIdx/blockIdx, so each thread does one
// element of the array. That's the whole parallel model.
// ---------------------------------------------------------
__global__ void vectorAddKernel(const float* a, const float* b, float* out, int n) {
    // Compute this thread's unique global index.
    // blockIdx.x  = which block this thread belongs to
    // blockDim.x  = how many threads per block
    // threadIdx.x = this thread's index within its block
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: total threads launched is usually rounded up to a
    // multiple of block size, so some threads at the end may
    // have i >= n. Without this check they'd read/write out of bounds.
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20; // ~1 million elements
    const size_t bytes = N * sizeof(float);

    // --- Host (CPU) memory: plain std::vector, nothing special ---
    std::vector<float> h_a(N), h_b(N), h_out(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // --- Device (GPU) memory: raw pointers, allocated with cudaMalloc ---
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    // --- Copy inputs from host to device ---
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // --- Launch configuration ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // ceil division

    // --- Launch the kernel ---
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_out, N);

    // Kernel launches are asynchronous — block until GPU finishes
    cudaDeviceSynchronize();

    // Always check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // --- Copy result back from device to host ---
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // --- Verify correctness against expected values ---
    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_out[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << h_out[i]
                      << ", expected " << expected << std::endl;
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "PASS" : "FAIL") << ": vector add of "
              << N << " elements." << std::endl;

    // --- Clean up device memory ---
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}