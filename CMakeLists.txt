cmake_minimum_required(VERSION 3.18)
project(cuda_dl_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 4060 laptop = Ada Lovelace = compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(vector_add src/vector_add.cu)

target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(matmul_naive src/matmul_naive.cu)

target_compile_options(matmul_naive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(matmul_tiled src/matmul_tiled.cu)

target_compile_options(matmul_tiled PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(relu_backward src/relu_backward.cu)

target_compile_options(relu_backward PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(matmul_backward src/matmul_backward.cu)

target_compile_options(matmul_backward PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(autograd_test src/autograd_test.cu)

target_compile_options(autograd_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(train_mnist src/train_mnist.cu)

target_compile_options(train_mnist PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
