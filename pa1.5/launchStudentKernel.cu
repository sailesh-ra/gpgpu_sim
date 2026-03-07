#include <cuda.h>
#include <cuda_fp16.h>
#include "mma_intrinsics.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
// TODO: Implement this function...
// This is for grading. You will use this function with the testbench we provide.
// You can add more functions etc. here if you want.
// You only need to launch your kernel inside this function, everything else will be managed by the testbench.
// M, N, K are matrix dimensions
// A is row major, B is column major, C is row major
// A, B and C are pointers to the matrices

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

void launchStudentKernel(int M, int N, int K, __half* A,
                                              __half* B,
                                              float* C) {


  // Launch your kernel here with appropriate grid and block sizes...

  assert(M > 0 && N > 0 && K > 0);

  dim3 blockDim(256);
  dim3 gridDim(N / 32, M / 32);
    
  tensorcore_gemm<<<gridDim, blockDim>>>(A, B, C, M, N, K);

  cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

}
