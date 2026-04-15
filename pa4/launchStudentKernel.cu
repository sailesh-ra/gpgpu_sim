#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "mma_intrinsics.cuh"
#include <cooperative_groups.h>
#include <cuda/pipeline>

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

    extern __shared__ __align__(128) int8_t smem[];

    using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, 3>;
    __shared__ pipeline_state_t pipeline_state;

    __half *A_smem = (__half*)(smem);
    __half *B_smem = (__half*)(smem + 3 * 64 * 64 * sizeof(__half));
    float  *C_smem = (float*) (smem + 6 * 64 * 64 * sizeof(__half));

    __half *A_stage[3] = { A_smem, A_smem + 64*64, A_smem + 2*64*64 };
    __half *B_stage[3] = { B_smem, B_smem + 64*64, B_smem + 2*64*64 };

    for (int i = tid; i < 64*64; i += 128)
        C_smem[i] = 0.0f;
    __syncthreads();

    auto tb       = cooperative_groups::this_thread_block();
    auto pipeline = cuda::make_pipeline(tb, &pipeline_state);

    int num_batches = K / 64;

    // PROLOG stage 0
    pipeline.producer_acquire();
    for (int row = tid; row < 64; row += 128)
        cuda::memcpy_async(&A_stage[0][row*64], &A[(out_row+row)*K + 0],
                           cuda::aligned_size_t<16>(64*sizeof(__half)), pipeline);
    for (int col = tid; col < 64; col += 128)
        cuda::memcpy_async(&B_stage[0][col*64], &B[0 + (out_col+col)*K],
                           cuda::aligned_size_t<16>(64*sizeof(__half)), pipeline);
    pipeline.producer_commit();

    // PROLOG stage 1
    if (num_batches > 1) {
        pipeline.producer_acquire();
        for (int row = tid; row < 64; row += 128)
            cuda::memcpy_async(&A_stage[1][row*64], &A[(out_row+row)*K + 64],
                               cuda::aligned_size_t<16>(64*sizeof(__half)), pipeline);
        for (int col = tid; col < 64; col += 128)
            cuda::memcpy_async(&B_stage[1][col*64], &B[64 + (out_col+col)*K],
                               cuda::aligned_size_t<16>(64*sizeof(__half)), pipeline);
        pipeline.producer_commit();
    }

    // MAIN LOOP
    #pragma unroll 4
    for (int batch = 2; batch < num_batches; batch++) {
        int copy_idx    = batch % 3;
        int compute_idx = (batch - 2) % 3;
        int k_base      = batch * 64;

        pipeline.producer_acquire();
        for (int row = tid; row < 64; row += 128)
            cuda::memcpy_async(&A_stage[copy_idx][row*64],
                               &A[(out_row+row)*K + k_base],
                               cuda::aligned_size_t<16>(64*sizeof(__half)), pipeline);
        for (int col = tid; col < 64; col += 128)
            cuda::memcpy_async(&B_stage[copy_idx][col*64],
                               &B[k_base + (out_col+col)*K],
                               cuda::aligned_size_t<16>(64*sizeof(__half)), pipeline);
        pipeline.producer_commit();

        pipeline.consumer_wait();
        mma_m16n8k16_f16_f16_smem_row_col_64x64(A_stage[compute_idx], B_stage[compute_idx], C_smem);
        pipeline.consumer_release();
    }

    // EPILOG: drain stage num_batches-2
    pipeline.consumer_wait();
    mma_m16n8k16_f16_f16_smem_row_col_64x64(A_stage[(num_batches-2)%3],
                                              B_stage[(num_batches-2)%3], C_smem);
    pipeline.consumer_release();

    // EPILOG: drain stage num_batches-1
    if (num_batches > 1) {
        pipeline.consumer_wait();
        mma_m16n8k16_f16_f16_smem_row_col_64x64(A_stage[(num_batches-1)%3],
                                                  B_stage[(num_batches-1)%3], C_smem);
        pipeline.consumer_release();
    }

    __syncthreads();

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