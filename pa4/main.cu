// Standard libraries you might need. You can add/remove as you like.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#pragma nv_diag_suppress static_var_with_dynamic_init
#include "mma_intrinsics.cuh"
#include <cassert>
#include <cooperative_groups.h>
#include <cuda/pipeline>

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

    int tid     = threadIdx.x;
    int out_row = blockIdx.y * 64;
    int out_col = blockIdx.x * 64;

    extern __shared__ __align__(128) int8_t smem[];

    // Block-scoped pipeline shared state (2 stages) in shared memory
    using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;
    __shared__ pipeline_state_t pipeline_state;

    __half *A_smem = (__half*)(smem);
    __half *B_smem = (__half*)(smem + 2 * 64 * 64 * sizeof(__half));
    float  *C_smem = (float*) (smem + 4 * 64 * 64 * sizeof(__half));

    __half *A_stage[2] = { A_smem, A_smem + 64*64 };
    __half *B_stage[2] = { B_smem, B_smem + 64*64 };

    for (int i = tid; i < 64*64; i += 128)
        C_smem[i] = 0.0f;
    __syncthreads();

    auto tb       = cooperative_groups::this_thread_block();
    auto pipeline = cuda::make_pipeline(tb, &pipeline_state);

    int num_batches = K / 64;

    // PROLOG
    pipeline.producer_acquire();
    // A: 64 rows, each row = 64 halfs = 128 bytes, contiguous in global (row-major)
    for (int row = tid; row < 64; row += 128) {
        cuda::memcpy_async(
            &A_stage[0][row * 64],
            &A[(out_row + row) * K + 0],           // k_base=0 for prolog
            cuda::aligned_size_t<16>(64 * sizeof(__half)),  // 128 bytes at once
            pipeline
        );
    }

    // B: 64 cols, each col = 64 halfs = 128 bytes, contiguous in global (col-major)
    for (int col = tid; col < 64; col += 128) {
        cuda::memcpy_async(
            &B_stage[0][col * 64],
            &B[0 + (out_col + col) * K],           // k_base=0 for prolog
            cuda::aligned_size_t<16>(64 * sizeof(__half)),
            pipeline
        );
    }

    pipeline.producer_commit();

    // MAIN LOOP — no __syncthreads() needed inside
    for (int batch = 1; batch < num_batches; batch++) {
        int copy_idx    = batch % 2;
        int compute_idx = (batch - 1) % 2;
        int k_base      = batch * 64;

        pipeline.producer_acquire();
        for (int row = tid; row < 64; row += 128) {
            cuda::memcpy_async(
                &A_stage[copy_idx][row * 64],
                &A[(out_row + row) * K + k_base],
                cuda::aligned_size_t<16>(64 * sizeof(__half)),
                pipeline
            );
        }
        for (int col = tid; col < 64; col += 128) {
            cuda::memcpy_async(
                &B_stage[copy_idx][col * 64],
                &B[k_base + (out_col + col) * K],
                cuda::aligned_size_t<16>(64 * sizeof(__half)),
                pipeline
            );
        }
        pipeline.producer_commit();

        pipeline.consumer_wait();
        mma_m16n8k16_f16_f16_smem_row_col_64x64(A_stage[compute_idx], B_stage[compute_idx], C_smem);
        pipeline.consumer_release();
    }

    // EPILOG
    pipeline.consumer_wait();
    mma_m16n8k16_f16_f16_smem_row_col_64x64(A_stage[(num_batches-1) % 2],
                                              B_stage[(num_batches-1) % 2],
                                              C_smem);
    pipeline.consumer_release();

    __syncthreads();

    for (int i = tid; i < 64*64; i += 128) {
        int row = i / 64, col = i % 64;
        int gr = out_row + row, gn = out_col + col;
        if (gr < M && gn < N)
            C[gr * N + gn] = C_smem[i];
    }
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

    size_t sizeA = static_cast<size_t>(M) * K;
    size_t sizeB = static_cast<size_t>(K) * N;
    size_t sizeC = static_cast<size_t>(M) * N;

    std::vector<__half> hA(sizeA);
    std::vector<__half> hB(sizeB);
    std::vector<float>  hC(sizeC, 0.0f);

    // Initialize A row-major
    for (size_t i = 0; i < sizeA; i++) {
        float val = static_cast<float>((i % 7) - 3) * 0.5f;
        hA[i] = __float2half(val);
    }

    // Initialize B column-major: element (k,n) at hB[k + n*K]
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            float val = static_cast<float>(((k + n) % 5) - 2) * 0.25f;
            hB[k + n * K] = __float2half(val);
        }
    }

    __half *dA = nullptr, *dB = nullptr;
    float  *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

    dim3 block(128, 1, 1);
    dim3 grid(N / 64, M / 64);

    int smemBytes = 65536;
    CHECK_CUDA(cudaFuncSetAttribute(tensorcore_gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

    std::cout << "Launching grid=(" << grid.x << "," << grid.y
              << "), block=(" << block.x << ")\n";

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    tensorcore_gemm<<<grid, block, smemBytes>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int runs = 100;

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        tensorcore_gemm<<<grid, block, smemBytes>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= runs;

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    double flops  = 2.0 * static_cast<double>(M) * N * K;
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
