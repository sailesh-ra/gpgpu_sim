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

__global__ void tensorcore_gemm(__half *A, __half *B, float *C, int M, int N, int K){

    int warpID  = threadIdx.x / 32;
    int laneID  = threadIdx.x % 32;

    int warp_row = warpID / 4;
    int warp_col = warpID % 4;

    int out_row = blockIdx.y * 32 + warp_row * 16;
    int out_col = blockIdx.x * 32 + warp_col * 8;

    __shared__ __half A_shared[8*256];
    __shared__ __half B_shared[8*128];
    __shared__ float  C_shared[8*128];

    // Each warp gets a pointer to its own slice
    __half *warp_A = A_shared + warpID * 256;
    __half *warp_B = B_shared + warpID * 128;
    float  *warp_C = C_shared + warpID * 128;

    // Initialize C_shared to 0
    for (int i = laneID; i < 128; i += 32)
        warp_C[i] = 0.0f;
    __syncthreads();

    // K loop
    for (int iter = 0; iter < K/16; iter++) {

        // Load A (row-major global → row-major shared)
        for (int i = laneID; i < 256; i += 32) {
            int row = i / 16;
            int col = i % 16;
            int gr  = out_row + row;
            int gk  = iter * 16 + col;
            warp_A[i] = (gr < M && gk < K) ? A[gr * K + gk] : __float2half(0.f);
        }

        // Load B (row-major global → col-major shared)
        for (int i = laneID; i < 128; i += 32) {
            int col = i / 16;
            int row = i % 16;
            int gk  = iter * 16 + row;
            int gn  = out_col + col;
            warp_B[col * 16 + row] = (gk < K && gn < N) ? B[gk * N + gn] : __float2half(0.f);
        }

        __syncthreads();
        mma_m16n8k16_f16_f16_smem_row_col(warp_A, warp_B, warp_C);
        __syncthreads();
    }

    // Store C back to global
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
