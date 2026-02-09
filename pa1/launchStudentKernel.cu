#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
// TODO: Implement this function...
// This is for grading. You will use this function with the testbench we provide.
// You can add more functions etc. here if you want.
// You only need to launch your kernel inside this function, everything else will be managed by the testbench.
// M, N, K are matrix dimensions
// layoutA and layoutB are memory layouts
// 0-> RowMajor
// 1-> ColumnMajor
// A, B and C are pointers to the matrices

//Naive GEMM kernel : One thread computes one C(row,col) element
__global__ void gemmNaiveKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                int layout_A, int layout_B)
{
    // Map thread to a single C element
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    // Dot product over K
    for(int k = 0; k < K; k++){
      // A is MxK, index A(row,k)
      float a_val = (layout_A == 0) ? A[row * K + k] // row-major
                                    : A[k * M + row] // col-major

      // B is KxN, index B(k,col)
      float b_val = (layout_B == 0) ? B[k * N + col] // row-major
                                    : B[col * K + k] // col-major
      
      acc += a_val * b_val;
    }

    // C is MxN row-major
    C[row * N + col] = acc;
}

void launchStudentKernel(int M, int N, int K, int layoutA,
                                              int layoutB,
                                              float* A,
                                              float* B,
                                              float* C) {

                                                

  // Launch your kernel here with appropriate grid and block sizes...

    // A good starter block size
    const int TILE_X = 16;
    const int TILE_Y = 16;

    dim3 block(TILE_X,TILE_Y);
    dim3 grid((N + TILE_X - 1) / TILE_X,
              (M + TILE_Y) / TILE_Y);

    gemmNaiveKernel<<<grid, block>>>(A,B,C,M,N,K,layout_A,layout_B);

    // catch launch errors eaarly during debugging
    // remove before submission

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
