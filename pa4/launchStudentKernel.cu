#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "mma_intrinsics.cuh"

__global__ void tensorcore_gemm(__half *A, __half *B, float *C, int M, int N, int K) {

    int tid     = threadIdx.x;
    int out_row = blockIdx.y * 64;
    int out_col = blockIdx.x * 64;

    // smem layout: [ A_ping | A_pong | B_ping | B_pong | C_tile ]
    //   A/B stages : 2 × 64×64 × sizeof(__half) = 16384 bytes each
    //   C tile     : 64×64 × sizeof(float)       = 16384 bytes
    //   Total      : 49152 bytes < 65536 ✓
    extern __shared__ __align__(128) int8_t smem[];

    __half *A_smem = (__half*)(smem);
    __half *B_smem = (__half*)(smem + 2 * 64 * 64 * sizeof(__half));
    float  *C_smem = (float*) (smem + 4 * 64 * 64 * sizeof(__half));

    __half *A_stage[2] = { A_smem, A_smem + 64*64 };
    __half *B_stage[2] = { B_smem, B_smem + 64*64 };

    // Initialize C tile in shared memory to 0
    for (int i = tid; i < 64*64; i += 128)
        C_smem[i] = 0.0f;
    __syncthreads();

    int num_batches = K / 64;
    auto pipeline = cuda::make_pipeline();

    // ── PROLOG: load batch 0 → stage 0 ───────────────────────────
    pipeline.producer_acquire();
    for (int i = tid; i < 64*64; i += 128) {
        int row = i / 64, col = i % 64;
        cuda::memcpy_async(&A_stage[0][i],
                           &A[(out_row + row) * K + col],
                           cuda::aligned_size_t<2>(sizeof(__half)), pipeline);
    }
    for (int i = tid; i < 64*64; i += 128) {
        int col = i / 64, row = i % 64;
        cuda::memcpy_async(&B_stage[0][col * 64 + row],
                           &B[row + (out_col + col) * K],
                           cuda::aligned_size_t<2>(sizeof(__half)), pipeline);
    }
    pipeline.producer_commit();

    // ── MAIN LOOP: batch 1 .. num_batches-1 ──────────────────────
    for (int batch = 1; batch < num_batches; batch++) {
        int copy_idx    = batch % 2;
        int compute_idx = (batch - 1) % 2;
        int k_base      = batch * 64;

        pipeline.producer_acquire();
        for (int i = tid; i < 64*64; i += 128) {
            int row = i / 64, col = i % 64;
            cuda::memcpy_async(&A_stage[copy_idx][i],
                               &A[(out_row + row) * K + (k_base + col)],
                               cuda::aligned_size_t<2>(sizeof(__half)), pipeline);
        }
        for (int i = tid; i < 64*64; i += 128) {
            int col = i / 64, row = i % 64;
            cuda::memcpy_async(&B_stage[copy_idx][col * 64 + row],
                               &B[(k_base + row) + (out_col + col) * K],
                               cuda::aligned_size_t<2>(sizeof(__half)), pipeline);
        }
        pipeline.producer_commit();

        pipeline.consumer_wait();
        __syncthreads();
        mma_m16n8k16_f16_f16_smem_row_col_64x64(A_stage[compute_idx], B_stage[compute_idx], C_smem);
        pipeline.consumer_release();
    }

    // ── EPILOG: consume last tile ─────────────────────────────────
    pipeline.consumer_wait();
    __syncthreads();
    mma_m16n8k16_f16_f16_smem_row_col_64x64(A_stage[(num_batches-1) % 2],
                                              B_stage[(num_batches-1) % 2],
                                              C_smem);
    pipeline.consumer_release();

    // All warps done writing C_smem before any thread reads it
    __syncthreads();

    // ── Store C_smem → global C (row-major) ──────────────────────
    for (int i = tid; i < 64*64; i += 128) {
        int row = i / 64, col = i % 64;
        int gr = out_row + row, gn = out_col + col;
        if (gr < M && gn < N)
            C[gr * N + gn] = C_smem[i];
    }
}

void launchStudentKernel(int M, int N, int K, __half* A, __half* B, float* C) {
    dim3 grid(N / 64, M / 64);
    dim3 block(128, 1, 1);

    int smemBytes = 65536;
    cudaFuncSetAttribute(tensorcore_gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);

    tensorcore_gemm<<<grid, block, smemBytes>>>(A, B, C, M, N, K);
}