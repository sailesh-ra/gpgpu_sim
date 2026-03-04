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

    __shared__ __half A_shared[256];
    __shared__ __half B_shared[128];
    __shared__ float  C_shared[128];

    // Initialize C_shared to 0
    for (int i = laneID; i < 128; i += 32)
        C_shared[i] = 0.0f;
    __syncwarp();

    // K loop
    for (int iter = 0; iter < K/16; iter++) {

        // Load A (row-major)
        for (int i = laneID; i < 256; i += 32) {
            int row = i / 16;
            int col = i % 16;
            A_shared[i] = A[(out_row + row) * K + (iter * 16 + col)];
        }

        // Load B (col-major)
        for (int i = laneID; i < 128; i += 32) {
            int col = i / 16;
            int row = i % 16;
            B_shared[i] = B[(iter * 16 + row) + (out_col + col) * K];
        }

        __syncwarp();
        mma_m16n8k16_f16_f16_smem_row_col(A_shared, B_shared, C_shared);
    }

    // Store C back to global
    for (int i = laneID; i < 128; i += 32) {
        int row = i / 8;
        int col = i % 8;
        C[(out_row + row) * N + (out_col + col)] = C_shared[i];
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
