// Standard libraries you might need. You can add/remove as you like.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>

void launchStudentKernel(int M, int N, int K,int layoutA, int layoutB,float* A, float* B, float* C);


static bool read_matrix_txt(const std::string& path,
                            std::vector<float>& out,
                            int& layout, int& rows, int& cols)
{
    std::ifstream in(path);
    if (!in) return false;

    if (!(in >> layout)) return false;
    if (!(in >> rows))   return false;
    if (!(in >> cols))   return false;

    out.resize(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        if (!(in >> out[i])) return false;
    }
    return true;
}
  
  static void compare_C(const std::vector<float>& got,
                        const std::vector<float>& ref,
                        int M, int N,
                        float atol = 1e-3f, float rtol = 1e-3f)
  {
    int bad = 0;
    float max_abs = 0.0f;
    int max_i = -1;

    for (int i = 0; i < M * N; i++) {
      float a = got[i];
      float b = ref[i];
      float diff = std::fabs(a-b);
      float tol = atol + rtol * std::fabs(b);

      if(diff > tol) {
          if(bad < 10) {
          int r = i / N, c = i % N;
          std::cout << "Mismatch at ("<< r << ","<< c << "): got = " << a
                    << "ref = " << b << " diff =" << diff << "tol = " << "\n";
        }
        bad++;
      }
      if (diff > max_abs) {
        max_abs = diff;
        max_i = i;
      }
    }
    if (bad == 0) {
      std::cout << "[SUCCESS] Output matches reference (within tolerance)\n";
    } else {
        int r = max_i / N, c = max_i % N;
        std::cout << "[FAIL] mismatches = " << bad
                  << " max_abs_diff = " << max_abs
                  << " at (" << r << "," << c << ")\n";
    }
  }

// This is where you allocate GPU memory, launch kernels and measure timings. 
// You'll need this when writing your report.
int main(int argc, char* argv[]) {

  std::string folder_path = "sample_1";
  std::string A_path = folder_path + "/A_64x64_T.txt";
  std::string B_path = folder_path + "/B_64x64_T.txt";
  std::string C_path = folder_path + "/C_64x64_T.txt";

  std::vector<float> h_A,h_B,h_C_ref;

  int layout_A, Arows, Acols;
  int layout_B, Brows, Bcols;
  int layout_C, Crows, Ccols;

  if(!read_matrix_txt(A_path,h_A,layout_A, Arows, Acols)) {
    std::cerr << "Failed to read A or wrong no. of elements: " << A_path << "\n";
    return 1;
  }
  if(!read_matrix_txt(B_path,h_B,layout_B, Brows, Bcols)) {
    std::cerr << "Failed to read B or wrong no. of elements: " << B_path << "\n";
    return 1;
  }
  if(!read_matrix_txt(C_path,h_C_ref,layout_C, Crows, Ccols)) {
    std::cerr << "Failed to read C or wrong no. of elements: " << C_path << "\n";
    return 1;
  }
  // cudaMalloc()...
  // Launch kernel and time it...

    // Infer GEMM dims: A is MxK, B is KxN, C is MxN
    int M = Arows;
    int K = Acols; // or Brows
    int N = Bcols;

    // Sanity checks
    if (Brows != K) {
        std::cerr << "Dim mismatch: A is " << Arows << "x" << Acols
                  << " but B is " << Brows << "x" << Bcols << "\n";
        return 1;
    }
    if (Crows != M || Ccols != N) {
        std::cerr << "Dim mismatch: C is " << Crows << "x" << Ccols
                  << " but expected " << M << "x" << N << "\n";
        return 1;
    }
    
    if ((int)h_A.size() != Arows*Acols || (int)h_B.size() != Brows*Bcols || (int)h_C_ref.size() != Crows*Ccols) {
    std::cerr << "Data size mismatch vs header.\n";
    return 1;
    }

  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_B, sizeB);
  cudaMalloc(&d_C, sizeC);

  cudaMemcpy(d_A,h_A.data(), sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,h_B.data(), sizeB, cudaMemcpyHostToDevice);

  launchStudentKernel(M, N, K,layout_A, layout_B,d_A, d_B, d_C);
  cudaDeviceSynchronize();

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
