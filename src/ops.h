#pragma once
#include "tensor.h"

// --- forward kernel (same one you've already written and verified) ---
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

// --- backward kernels (same ones you've already written and verified) ---
__global__ void matmulBackwardAKernel(const float* dC, const float* B, float* dA, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && k < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            sum += dC[row * N + n] * B[k * N + n];
        }
        dA[row * K + k] += sum; // note: += not =, since gradients accumulate
    }
}

__global__ void matmulBackwardBKernel(const float* A, const float* dC, float* dB, int M, int K, int N) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K && col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) {
            sum += A[m * K + k] * dC[m * N + col];
        }
        dB[k * N + col] += sum; // note: += not =
    }
}

inline Tensor* matmulForward(Tensor* A, Tensor* B) {
    int M = A->rows, K = A->cols, N = B->cols;

    Tensor* C = new Tensor(M, N);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
    matmulForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(A->data, B->data, C->data, M, K, N);
    cudaDeviceSynchronize();

    tape.push_back([A, B, C, M, K, N]() {
        dim3 threadsPerBlock(16, 16);

        dim3 gridA((K + 15) / 16, (M + 15) / 16);
        matmulBackwardAKernel<<<gridA, threadsPerBlock>>>(C->grad, B->data, A->grad, M, K, N);

        dim3 gridB((N + 15) / 16, (K + 15) / 16);
        matmulBackwardBKernel<<<gridB, threadsPerBlock>>>(A->data, C->grad, B->grad, M, K, N);

        cudaDeviceSynchronize();
    });

    return C;
}

__global__ void reluForwardKernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

__global__ void reluBackwardKernel(const float* x, const float* dL_dy, float* dL_dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dL_dx[i] += (x[i] > 0.0f) ? dL_dy[i] : 0.0f; // += now, same reasoning as matmul
    }
}

inline Tensor* reluForward(Tensor* x) {
    Tensor* y = new Tensor(x->rows, x->cols);
    int n = x->size();

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    reluForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(x->data, y->data, n);
    cudaDeviceSynchronize();

    tape.push_back([x, y, n]() {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        reluBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(x->data, y->grad, x->grad, n);
        cudaDeviceSynchronize();
    });

    return y;
}

__global__ void biasAddForwardKernel(const float* x, const float* b, float* y, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        y[row * cols + col] = x[row * cols + col] + b[col];
    }
}

__global__ void biasAddBackwardKernel(const float* dy, float* db, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += dy[row * cols + col];
        }
        db[col] += sum;
    }
}

__global__ void addInto(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] += src[i];
    }
}

__global__ void softmaxCrossEntropyForwardKernel(const float* logits, const float* labelsOneHot,
                                                   float* probs, float* lossPerRow,
                                                   int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        // 1. find max logit in this row, for numerical stability
        float maxLogit = logits[row * cols];
        for (int c = 1; c < cols; c++) {
            maxLogit = fmaxf(maxLogit, logits[row * cols + c]);
        }

        // 2. compute exp(logit - max) for each class, and their sum
        float sumExp = 0.0f;
        for (int c = 0; c < cols; c++) {
            float e = expf(logits[row * cols + c] - maxLogit);
            probs[row * cols + c] = e;
            sumExp += e;
        }

        // 3. normalize into actual probabilities
        for (int c = 0; c < cols; c++) {
            probs[row * cols + c] /= sumExp;
        }

        // 4. cross-entropy loss for this row: -log(probability assigned to the true class)
        float loss = 0.0f;
        for (int c = 0; c < cols; c++) {
            if (labelsOneHot[row * cols + c] > 0.5f) {
                loss = -logf(probs[row * cols + c] + 1e-8f); // tiny epsilon avoids log(0)
            }
        }
        lossPerRow[row] = loss;
    }
}

__global__ void softmaxCrossEntropyBackwardKernel(const float* probs, const float* labelsOneHot,
                                                    float* dLogits, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        dLogits[row * cols + col] += (probs[row * cols + col] - labelsOneHot[row * cols + col]) / rows;
    }
}

__global__ void sgdUpdateKernel(float* param, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        param[i] -= lr * grad[i];
    }
}


inline Tensor* biasAddForward(Tensor* x, Tensor* b) {
    int rows = x->rows, cols = x->cols;
    Tensor* y = new Tensor(rows, cols);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);
    biasAddForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(x->data, b->data, y->data, rows, cols);
    cudaDeviceSynchronize();

    tape.push_back([x, b, y, rows, cols]() {
        // dx = dy unchanged, so just accumulate directly into x->grad
        int n = rows * cols;
        int threadsPerBlock1D = 256;
        int blocksPerGrid1D = (n + threadsPerBlock1D - 1) / threadsPerBlock1D;
        // reuse a simple elementwise add-into kernel for dx (defined below)
        addInto<<<blocksPerGrid1D, threadsPerBlock1D>>>(x->grad, y->grad, n);

        int threadsPerBlockB = 256;
        int blocksPerGridB = (cols + threadsPerBlockB - 1) / threadsPerBlockB;
        biasAddBackwardKernel<<<blocksPerGridB, threadsPerBlockB>>>(y->grad, b->grad, rows, cols);

        cudaDeviceSynchronize();
    });

    return y;
}

inline float softmaxCrossEntropyLoss(Tensor* logits, Tensor* labelsOneHot) {
    int rows = logits->rows, cols = logits->cols;

    float* probs;
    float* lossPerRow;
    cudaMalloc(&probs, rows * cols * sizeof(float));
    cudaMalloc(&lossPerRow, rows * sizeof(float));

    int threadsPerBlock1D = 256;
    int blocksPerGrid1D = (rows + threadsPerBlock1D - 1) / threadsPerBlock1D;
    softmaxCrossEntropyForwardKernel<<<blocksPerGrid1D, threadsPerBlock1D>>>(
        logits->data, labelsOneHot->data, probs, lossPerRow, rows, cols);
    cudaDeviceSynchronize();

    // pull loss values back to host and average them into a single number
    std::vector<float> h_lossPerRow(rows);
    cudaMemcpy(h_lossPerRow.data(), lossPerRow, rows * sizeof(float), cudaMemcpyDeviceToHost);
    float totalLoss = 0.0f;
    for (int i = 0; i < rows; i++) totalLoss += h_lossPerRow[i];
    float meanLoss = totalLoss / rows;

    // immediately compute the starting gradient (no need to wait/push to tape —
    // this IS the start of backward, since loss has nothing downstream of it)
    dim3 threadsPerBlock2D(16, 16);
    dim3 blocksPerGrid2D((cols + 15) / 16, (rows + 15) / 16);
    softmaxCrossEntropyBackwardKernel<<<blocksPerGrid2D, threadsPerBlock2D>>>(
        probs, labelsOneHot->data, logits->grad, rows, cols);
    cudaDeviceSynchronize();

    cudaFree(probs);
    cudaFree(lossPerRow);

    return meanLoss;
}

inline void sgdUpdate(Tensor* param, float lr) {
    int n = param->size();
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    sgdUpdateKernel<<<blocksPerGrid, threadsPerBlock>>>(param->data, param->grad, lr, n);
    cudaDeviceSynchronize();

    cudaMemset(param->grad, 0, n * sizeof(float)); // reset gradient for the next training step
}
