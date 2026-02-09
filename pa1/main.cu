// Standard libraries you might need. You can add/remove as you like.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <iomanip>
#include <vector>

// This is where you allocate GPU memory, launch kernels and measure timings. 
// You'll need this when writing your report.
int main(int argc, char* argv[]) {

  // Sample Size (from sample inputs and outputs folder)
  int M = 64, N = 64, K = 64;

  // 0 = row-major, 1 = col-major
  int layout_A = 0;
  int layout_B = 1;

  std::string A_path = "sample_0/A_64x64_T.txt"
  std::string B_path = "sample_0/B_64x64_T.txt"
  std::string C_path = "sample_0/C_64x64_T.txt"

  std::vector<float> h_A,h_B,h_C_ref;

  if(!read_matrix_text(A_path,h_A,M,K)) {
    std::cerr << "Failed to read A or wrong no. of elements: " << A_path << "\n";
    return 1;
  }
  if(!read_matrix_text(B_path,h_B,M,K)) {
    std::cerr << "Failed to read B or wrong no. of elements: " << B_path << "\n";
    return 1;
  }
  if(!read_matrix_text(C_path,h_C_ref,M,N)) {
    std::cerr << "Failed to read C or wrong no. of elements: " << C_path << "\n";
    return 1;
  }
  // cudaMalloc()...
  // Launch kernel and time it...

  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_B, sizeB);
  cudaMalloc(&d_C, sizeC);

  cudamemcpy(d_a,h_a.data(), sizeA, cudaMemcpyHostToDevice);
  cudamemcpy(d_b,h_b.data(), sizeB, cudaMemcpyHostToDevice);

  launchStudentKernel(d_A, d_B, d_C, M, N, K, layout_A, layout_B);

  std::vector<float> h_C(M * N);
  cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Compare
  compare_C(h_C, h_C_ref, M, N);

  // cudaMallocManaged()
  // Launch kernel and time it...

  return 0;
}
