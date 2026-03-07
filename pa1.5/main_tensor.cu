// Standard libraries you might need. You can add/remove as you like.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "mma_intrinsics.cuh"
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

__global__ void tensorcore_gemm(__half *A, __half *B, float *C, int M, int N, int K) {
    int warpID  = threadIdx.x / 32;
    int laneID  = threadIdx.x % 32;

    int warp_row = warpID / 4;
    int warp_col = warpID % 4;

    int out_row = blockIdx.y * 32 + warp_row * 16;
    int out_col = blockIdx.x * 32 + warp_col * 8;

    __shared__ __half A_shared[2 * 256];
    __shared__ __half B_shared[4 * 128];
    __shared__ float  C_shared[8 * 128];

    __half *warp_A = A_shared + warp_row * 256;
    __half *warp_B = B_shared + warp_col * 128;
    float  *warp_C = C_shared + warpID * 128;

    for (int i = laneID; i < 128; i += 32)
        warp_C[i] = 0.0f;
    __syncthreads();

    for (int iter = 0; iter < K/16; iter++) {

        int k_base = iter * 16;

        if (warp_col == 0){
        #pragma unroll
        for (int i = laneID; i < 256; i += 32) {
            int row = i / 16;
            int col = i % 16;

            int gr = out_row + row;
            int gk = k_base + col;

            A_shared[warp_row * 256 + i] = A[gr * K + gk];
            }
        }
        if (warp_row == 0){
        #pragma unroll
        for (int i = laneID; i < 128; i += 32) {
            int col = i / 16;
            int row = i % 16;

            int gk = k_base + row;
            int gn = out_col + col;

            B_shared[warp_col * 128 + col * 16 + row] = B[gk + gn * K];;
        }
    }
        __syncthreads();
        mma_m16n8k16_f16_f16_smem_row_col(warp_A, warp_B, warp_C);
        __syncthreads();
        }

        for (int i = laneID; i < 128; i += 32) {
            int row = i / 8;
            int col = i % 8;
            int gr  = out_row + row;
            int gn  = out_col + col;
            if (gr < M && gn < N)
                C[gr * N + gn] = warp_C[i];
        }
    }

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

    size_t sizeA = static_cast<size_t>(M) * K;  // row-major MxK
    size_t sizeB = static_cast<size_t>(K) * N;  // column-major KxN in memory
    size_t sizeC = static_cast<size_t>(M) * N;  // row-major MxN

    std::vector<__half> hA(sizeA);
    std::vector<__half> hB(sizeB);
    std::vector<float>  hC(sizeC, 0.0f);

    // Initialize A row-major
    for (size_t i = 0; i < sizeA; i++) {
        float val = static_cast<float>((i % 7) - 3) * 0.5f;
        hA[i] = __float2half(val);
    }

    // Initialize B as column-major storage for logical KxN matrix
    // element (k,n) stored at hB[k + n*K]
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            float val = static_cast<float>(((k + n) % 5) - 2) * 0.25f;
            hB[k + n * K] = __float2half(val);
        }
    }

    __half *dA = nullptr;
    __half *dB = nullptr;
    float  *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

    dim3 block(256);
    dim3 grid(N/32, M/32);

    std::cout << "Launching grid=(" << grid.x << "," << grid.y
              << "), block=(" << block.x << ")\n";

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Optional warmup
    tensorcore_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int runs = 100;

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
    tensorcore_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= runs;

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

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
