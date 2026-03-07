// Standard libraries you might need. You can add/remove as you like.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>
#include <stdexcept>
#include <cassert>

#define CHECK_CUDA(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                  << " -> " << cudaGetErrorString(err) << std::endl;       \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
  } while (0)

//Naive GEMM kernel : One thread computes one C(row,col) element
__global__ void gemmNaiveKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                int layoutA, int layoutB)
{
    // Map thread to a single C element
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    // Dot product over K
    for(int k = 0; k < K; k++){
      // A is MxK, index A(row,k)
      float a_val = (layoutA == 0) ? A[row * K + k] // row-major
                                    : A[k * M + row]; // col-major

      // B is KxN, index B(k,col)
      float b_val = (layoutB == 0) ? B[k * N + col] // row-major
                                    : B[col * K + k]; // col-major
      
      acc += a_val * b_val;
    }

    // C is MxN row-major
    C[row * N + col] = acc;
}

// This is where you allocate GPU memory, launch kernels and measure timings. 
// You'll need this when writing your report.
int main(int argc, char* argv[]) {

  if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " M N K\n";
        return EXIT_FAILURE;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    assert(M > 0 && N > 0 && K > 0);

    std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;

    // Match your tensor timing workflow:
    // A row-major, B column-major, C row-major
    int layoutA = 0;
    int layoutB = 1;

    size_t elemsA = static_cast<size_t>(M) * K;
    size_t elemsB = static_cast<size_t>(K) * N;
    size_t elemsC = static_cast<size_t>(M) * N;

    size_t sizeA = elemsA * sizeof(float);
    size_t sizeB = elemsB * sizeof(float);
    size_t sizeC = elemsC * sizeof(float);

    std::vector<float> hA(elemsA);
    std::vector<float> hB(elemsB);
    std::vector<float> hC(elemsC, 0.0f);

    // Initialize A row-major
    for (size_t i = 0; i < elemsA; i++) {
        float val = static_cast<float>((i % 7) - 3) * 0.5f;
        hA[i] = val;
    }

    // Initialize B column-major: B(k,n) at hB[k + n*K]
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            float val = static_cast<float>(((k + n) % 5) - 2) * 0.25f;
            hB[k + n * K] = val;
        }
    }

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    std::cout << "Launching grid=(" << grid.x << "," << grid.y
              << "), block=(" << block.x << "," << block.y << ")\n";

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    gemmNaiveKernel<<<grid, block>>>(dA, dB, dC, M, N, K, layoutA, layoutB);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Choose more runs for smaller problems
    int runs = (M <= 512 && N <= 512 && K <= 512) ? 100 : 10;

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        gemmNaiveKernel<<<grid, block>>>(dA, dB, dC, M, N, K, layoutA, layoutB);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= runs;

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC, cudaMemcpyDeviceToHost));

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double tflops = flops / (elapsed_ms * 1e-3) / 1e12;

    std::cout << "Kernel time: " << elapsed_ms << " ms\n";
    std::cout << "Performance: " << tflops << " TFLOP/s\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}
