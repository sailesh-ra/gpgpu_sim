#include <cuda.h>
// TODO: Implement this function...
// This is for grading. You will use this function with the testbench we provide.
// You can add more functions etc. here if you want.
// You only need to launch your kernel inside this function, everything else will be managed by the testbench.
// M, N, K are matrix dimensions
// layoutA and layoutB are memory layouts
// 0-> RowMajor
// 1-> ColumnMajor
// A, B and C are pointers to the matrices
void launchStudentKernel(int M, int N, int K, int layoutA,
                                              int layoutB,
                                              float* A,
                                              float* B,
                                              float* C) {

  // Launch your kernel here with appropriate grid and block sizes...

}
