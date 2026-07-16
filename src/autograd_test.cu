#include "ops.h"
#include <iostream>

int main() {
    // Build: y = relu(A * B)
    Tensor* A = new Tensor(4, 4);
    Tensor* B = new Tensor(4, 4);

    // fill A and B with some simple host-side values via cudaMemcpy
    std::vector<float> h_A(16, 1.0f); // all 1s for a simple hand-checkable case
    std::vector<float> h_B(16, 1.0f);
    cudaMemcpy(A->data, h_A.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B->data, h_B.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor* C = matmulForward(A, B);   // forward pass #1
    Tensor* y = reluForward(C);        // forward pass #2

    // Fabricate an incoming gradient on the final output, as if from a loss function
    std::vector<float> h_dy(16, 1.0f);
    cudaMemcpy(y->grad, h_dy.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);

    backwardAll(); // walks the tape in reverse: relu backward, then matmul backward

    // Pull A's gradient back to host and print it
    std::vector<float> h_dA(16);
    cudaMemcpy(h_dA.data(), A->grad, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "dA (should be all 4.0, since each row of A=all-1s times a 4x4 all-1s B, "
              << "through a relu that's fully active, gives a clean derivative of 4 per element):" << std::endl;
    for (int i = 0; i < 16; i++) std::cout << h_dA[i] << " ";
    std::cout << std::endl;

    return 0;
}
