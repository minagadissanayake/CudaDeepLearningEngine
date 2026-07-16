# CUDA Deep Learning Engine

A minimal deep learning engine built from scratch in CUDA/C++ — including a hand-written tape-based autograd system — to understand what actually happens beneath frameworks like PyTorch at the memory and kernel level.

Trained on the full real MNIST dataset, this engine reaches **97.56% test accuracy** on unseen digits using only custom kernels for matrix multiplication, activation, bias addition, softmax cross-entropy loss, and SGD optimization — no external ML libraries.

## Why this project

GameVaultAI and Game Success Predictor (my two earlier projects) work at the application and model-training layer — calling APIs, training models with existing libraries. This project goes one level deeper: writing the actual compute kernels and the automatic differentiation system that libraries like PyTorch provide, to understand *why* GPU-accelerated deep learning works the way it does, not just how to call it.

## Architecture

```
Tensor (RAII wrapper around device memory + gradient buffer)
   │
   ├── matmulForward / matmulBackward   (naive + tiled/shared-memory variants)
   ├── reluForward / reluBackward
   ├── biasAddForward / biasAddBackward (with broadcasting)
   └── softmaxCrossEntropyLoss          (numerically stable, fused gradient)
   │
   ▼
Tape-based autograd
   — every forward op pushes a closure recording its own backward step
   — backwardAll() walks the tape in reverse, auto-chaining gradients
   │
   ▼
SGD optimizer (param -= lr * grad, then zero the gradient)
   │
   ▼
Training loop: 784 → 128 → 10 MLP, trained on MNIST
```

**Design choice — tape-based autograd over a full computation graph:** since the network here is a simple sequential chain (not branching), recording operations in forward order and walking the resulting list in reverse gets chain-rule ordering correct without needing a general graph/topological-sort structure. This is close to how early PyTorch's autograd worked before more general graph support was added.

## GPU environment

- RTX 4060 Laptop GPU (Ada Lovelace, compute capability 8.9)
- CUDA 12.0 (toolkit) via WSL2/Ubuntu
- Verified via `deviceQuery`: 49,152 bytes shared memory per block, 1,024 max threads per block

## Benchmark: naive vs. tiled (shared-memory) matmul

The project implements two matmul kernels to demonstrate a core GPU optimization technique: naive matmul re-reads the same rows/columns of the input matrices from slow global memory redundantly across threads; the tiled version has each block cooperatively stage small tiles of the inputs into fast on-chip shared memory once, then reuses them across all threads in the block.

| Matrix size | Naive (ms) | Tiled (ms) | Faster kernel |
|---|---|---|---|
| 256×256   | 0.41  | 0.94  | Naive (tiling overhead dominates at this size) |
| 1024×1024 | 3.34  | 2.88  | Tiled (~14% faster) |
| 2048×2048 | 20.80 | 16.23 | Tiled (~22% faster) |

**Finding:** tiling is not a free win — at small problem sizes, the synchronization overhead of shared-memory staging (two `__syncthreads()` barriers per tile) outweighs the memory-traffic savings, since the naive version's redundant reads are largely absorbed by cache anyway. The tiled kernel only pulls ahead once the matrix is large enough that global memory bandwidth, rather than kernel launch/sync overhead, becomes the actual bottleneck — and that advantage *grows* with problem size (14% faster at 1024³, 22% faster at 2048³), since larger matrices mean more redundant global-memory traffic in the naive version for tiling to eliminate. This crossover and its growth trend are well-documented properties of tiled GPU kernels, empirically reproduced here rather than assumed.

## Training result

2-layer MLP (784 → 128 → 10), trained on the full 60,000-example MNIST training set for 10 epochs, batch size 64, plain SGD (lr = 0.1):

| Epoch | Avg. loss |
|---|---|
| 0 | 0.419 |
| 1 | 0.216 |
| 2 | 0.162 |
| 3 | 0.130 |
| 4 | 0.109 |
| 5 | 0.094 |
| 6 | 0.082 |
| 7 | 0.073 |
| 8 | 0.066 |
| 9 | 0.060 |

**Test accuracy on the full 10,000-image held-out test set: 97.56% (9,740/9,984 correct).**

Loss decreases monotonically every epoch with no instability — evidence that the forward kernels, the hand-written backward kernels, the autograd tape, and the SGD update are all correctly wired together, not just individually correct in isolation. A 97.56% test accuracy on completely unseen data, using a plain fully-connected network with no convolutional layers, is a strong, competitive result for this architecture class.

## What's implemented

- Custom `Tensor` class: RAII-managed GPU memory (constructor allocates via `cudaMalloc`, destructor frees automatically)
- Naive and shared-memory-tiled matrix multiplication (forward + backward, including the transpose-via-indexing trick to avoid materializing transposed copies)
- ReLU forward/backward
- Bias addition with broadcasting (forward/backward, including gradient summation across the batch dimension)
- Numerically stable softmax + cross-entropy loss (max-subtraction trick), using the fused `probs - one_hot` gradient identity rather than separate softmax/cross-entropy backward passes
- A tape-based automatic differentiation system: every forward operation records a closure describing its own backward computation; `backwardAll()` replays these in reverse order
- SGD parameter updates
- End-to-end training and evaluation on real MNIST data (CSV-loaded, normalized, one-hot encoded)

## Known limitations / next steps

- **Softmax kernel is one-thread-per-row**, not a parallel reduction — fine at MNIST's scale (10 classes), but wouldn't scale to large vocabularies/class counts without a proper parallel reduction.
- **No convolutional layers** — the engine currently only supports fully-connected layers; adding a conv2d forward/backward kernel pair is the natural next extension.
- **Intermediate tensors use `new`/`delete` rather than a reference-counted or pooled allocator** — sufficient for this scale, but a real framework would need smarter memory management to avoid fragmentation at larger scale.
- **Single GPU, single stream** — no multi-GPU or asynchronous stream overlap.

## Tech stack

CUDA C++, CMake, WSL2/Ubuntu, RTX 4060 Laptop GPU (Ada Lovelace)
