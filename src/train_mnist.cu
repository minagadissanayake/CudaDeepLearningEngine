#include "ops.h"
#include "mnist_loader.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

const int INPUT_DIM = 784;   // 28x28 pixels flattened
const int HIDDEN_DIM = 128;
const int OUTPUT_DIM = 10;   // digits 0-9
const int BATCH_SIZE = 64;
const int NUM_EPOCHS = 10;
const float LEARNING_RATE = 0.1f;

// Fills a Tensor's device data with small random values (simple weight init)
void randomInit(Tensor* t, float scale) {
    std::vector<float> h(t->size());
    for (int i = 0; i < t->size(); i++) {
        h[i] = ((static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f) * scale;
    }
    cudaMemcpy(t->data, h.data(), t->size() * sizeof(float), cudaMemcpyHostToDevice);
}

int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    // --- Load MNIST ---
    std::vector<float> h_trainImages, h_trainLabels;
    loadMNIST("data/mnist_train.csv", 60000, h_trainImages, h_trainLabels);

    int numExamples = h_trainLabels.size() / OUTPUT_DIM; // total rows loaded

    // --- Network parameters ---
    Tensor* W1 = new Tensor(INPUT_DIM, HIDDEN_DIM);
    Tensor* b1 = new Tensor(1, HIDDEN_DIM);
    Tensor* W2 = new Tensor(HIDDEN_DIM, OUTPUT_DIM);
    Tensor* b2 = new Tensor(1, OUTPUT_DIM);

    randomInit(W1, 0.1f);
    randomInit(W2, 0.1f);
    // biases start at zero, which Tensor's constructor already does via cudaMemset — no need to randomInit these
    int numBatches = numExamples / BATCH_SIZE;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epochLoss = 0.0f;

        for (int b = 0; b < numBatches; b++) {
            // --- Slice out one batch's worth of images/labels from the loaded data ---
            Tensor* x = new Tensor(BATCH_SIZE, INPUT_DIM);
            Tensor* yTrue = new Tensor(BATCH_SIZE, OUTPUT_DIM);

            cudaMemcpy(x->data, &h_trainImages[b * BATCH_SIZE * INPUT_DIM],
                       BATCH_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(yTrue->data, &h_trainLabels[b * BATCH_SIZE * OUTPUT_DIM],
                       BATCH_SIZE * OUTPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);

            // --- Forward pass: x -> matmul -> bias -> relu -> matmul -> bias -> loss ---
            Tensor* z1 = matmulForward(x, W1);
            Tensor* a1 = biasAddForward(z1, b1);
            Tensor* h1 = reluForward(a1);
            Tensor* z2 = matmulForward(h1, W2);
            Tensor* logits = biasAddForward(z2, b2);

            float loss = softmaxCrossEntropyLoss(logits, yTrue);
            epochLoss += loss;

            // --- Backward pass: walk the tape in reverse ---
            backwardAll();

            // --- Update every trainable parameter ---
            sgdUpdate(W1, LEARNING_RATE);
            sgdUpdate(b1, LEARNING_RATE);
            sgdUpdate(W2, LEARNING_RATE);
            sgdUpdate(b2, LEARNING_RATE);

            // --- Clean up this batch's tensors (avoids unbounded memory growth) ---
            delete x;
            delete yTrue;
            delete z1;
            delete a1;
            delete h1;
            delete z2;
            delete logits;
        }

        std::cout << "Epoch " << epoch << " - avg loss: " << (epochLoss / numBatches) << std::endl;
    }

    // --- Test-set accuracy check ---
    std::vector<float> h_testImages, h_testLabels;
    loadMNIST("data/mnist_test.csv", 10000, h_testImages, h_testLabels);
    int numTestExamples = h_testLabels.size() / OUTPUT_DIM;

    int correct = 0;
    int testBatches = numTestExamples / BATCH_SIZE;

    for (int b = 0; b < testBatches; b++) {
        Tensor* x = new Tensor(BATCH_SIZE, INPUT_DIM);
        cudaMemcpy(x->data, &h_testImages[b * BATCH_SIZE * INPUT_DIM],
                   BATCH_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);

        // forward pass only -- no loss, no backward, no update
        Tensor* z1 = matmulForward(x, W1);
        Tensor* a1 = biasAddForward(z1, b1);
        Tensor* h1 = reluForward(a1);
        Tensor* z2 = matmulForward(h1, W2);
        Tensor* logits = biasAddForward(z2, b2);

        // pull predictions back to host and compare against true labels
        std::vector<float> h_logits(BATCH_SIZE * OUTPUT_DIM);
        cudaMemcpy(h_logits.data(), logits->data, BATCH_SIZE * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost);

        for (int row = 0; row < BATCH_SIZE; row++) {
            // find predicted class: index of the largest logit in this row
            int predicted = 0;
            float maxVal = h_logits[row * OUTPUT_DIM];
            for (int c = 1; c < OUTPUT_DIM; c++) {
                if (h_logits[row * OUTPUT_DIM + c] > maxVal) {
                    maxVal = h_logits[row * OUTPUT_DIM + c];
                    predicted = c;
                }
            }

            // find true class: index of the "1" in the one-hot label
            int trueLabel = 0;
            int globalRow = b * BATCH_SIZE + row;
            for (int c = 0; c < OUTPUT_DIM; c++) {
                if (h_testLabels[globalRow * OUTPUT_DIM + c] > 0.5f) {
                    trueLabel = c;
                    break;
                }
            }

            if (predicted == trueLabel) correct++;
        }

        // important: tape.clear() here, since we ran forward passes that pushed
        // backward closures we never intend to call -- otherwise they'd sit around
        // and get incorrectly triggered by a future backwardAll() call
        tape.clear();

        delete x;
        delete z1;
        delete a1;
        delete h1;
        delete z2;
        delete logits;
    }

    float accuracy = 100.0f * correct / (testBatches * BATCH_SIZE);
    std::cout << "Test accuracy: " << accuracy << "% (" << correct << "/"
              << testBatches * BATCH_SIZE << ")" << std::endl;

    delete W1;
    delete b1;
    delete W2;
    delete b2;

    return 0;
}
