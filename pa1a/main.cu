
// Uncomment exactly ONE of these
#define USE_CUDA_MALLOC
//#define USE_CUDA_MALLOC_MANAGED
//#define PREFETCH
//#define NO_PREFETCH

// Standard libraries you might need. You can add/remove as you like.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>
#include <stdexcept>
namespace fs = std::filesystem;

//Naive GEMM kernel : One thread computes one C(row,col) element
__global__ void gemmNaiveKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                int layoutA, int layoutB)
{
    // Map thread to a single C element
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    // Dot product over K
    for(int k = 0; k < K; k++){
      // A is MxK, index A(row,k)
      float a_val = (layoutA == 0) ? A[row * K + k] // row-major
                                    : A[k * M + row]; // col-major

      // B is KxN, index B(k,col)
      float b_val = (layoutB == 0) ? B[k * N + col] // row-major
                                    : B[col * K + k]; // col-major
      
      acc += a_val * b_val;
    }

    // C is MxN row-major
    C[row * N + col] = acc;
}


static bool read_matrix_txt(const std::string& path,
                            std::vector<float>& out,
                            int& layout, int& rows, int& cols)
{
    std::ifstream in(path);
    if (!in) return false;


    std::string layout_tok;
    if (!(in >> layout_tok)) return false;
    if (!(in >> rows))   return false;
    if (!(in >> cols))   return false;

    if (layout_tok == "T" || layout_tok == "t" || layout_tok == "0") layout = 0;      // row-major
    else if (layout_tok == "N" || layout_tok == "n" || layout_tok == "1") layout = 1; // col-major
    else return false;

    out.resize((size_t)rows * (size_t)cols);
    for (size_t i = 0; i < rows * cols; i++) {
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
                    << "ref = " << b << " diff =" << diff << "tol = " << tol << "\n";
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

  fs::path folder = "sample_3";

  fs::path A_path, B_path, C_path;

  for (const auto& entry : fs::directory_iterator(folder)) {
      std::string name = entry.path().filename().string();
      if (name.rfind("A_64x64_", 0) == 0) A_path = entry.path();
      if (name.rfind("B_64x64_", 0) == 0) B_path = entry.path();
      if (name.rfind("C_64x64_", 0) == 0) C_path = entry.path();
  }

  if (A_path.empty() || B_path.empty() || C_path.empty()) {
    throw std::runtime_error("Missing matrix file(s)");
  }

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

  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (M + blockDim.y - 1) / blockDim.y);

  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // cudaMalloc()...
  // Launch kernel and time it...

  #if defined (USE_CUDA_MALLOC)

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_B, sizeB);
  cudaMalloc(&d_C, sizeC);

  cudaMemcpy(d_A,h_A.data(), sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,h_B.data(), sizeB, cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, sizeC);


  cudaEventRecord(start);

  for (int i = 0; i < 100; i++){
    gemmNaiveKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K,layout_A, layout_B);
  } 

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= 100.0f;

  std::cout << "Kernel time (cudaMalloc) - iterations 1..99 (steady-state compute): " << ms << " ms\n";

  std::vector<float> h_C(M * N);
  cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Compare
  std::cout << "\n[UM] Comparing cudaMalloc result vs reference:\n";
  compare_C(h_C, h_C_ref, M, N);

  #endif

  // cudaMallocManaged()
  // Launch kernel and time it...

  #if defined(USE_CUDA_MALLOC_MANAGED)

  // ---- Unified Mem Version ----

  float *um_A = nullptr, *um_B = nullptr, *um_C = nullptr;

  cudaMallocManaged(&um_A,sizeA);
  cudaMallocManaged(&um_B,sizeB);
  cudaMallocManaged(&um_C,sizeC);

  // Init from host vectors
  std::memcpy(um_A, h_A.data(), sizeA);
  std::memcpy(um_B, h_B.data(), sizeB);

  // prefetching to avoid timing the first-touch page faults
  int dev = 0;
  cudaGetDevice(&dev);

  #if defined(NO_PREFETCH)

  std::memset(um_C, 0, sizeC);
  cudaDeviceSynchronize(); // make memset visible + start from clean state

  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    gemmNaiveKernel<<<gridDim, blockDim>>>(um_A, um_B, um_C, M, N, K, layout_A, layout_B);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms_no_pf = 0.0f;
  cudaEventElapsedTime(&ms_no_pf, start, stop);
  ms_no_pf /= 100.0f;

  std::cout << "UM kernel time (NO prefetch): " << ms_no_pf << " ms\n";

  #endif

  #if defined(PREFETCH)
  std::memset(um_C, 0, sizeC);

  cudaMemPrefetchAsync(um_A, sizeA, dev);
  cudaMemPrefetchAsync(um_B, sizeB, dev);
  cudaMemPrefetchAsync(um_C, sizeC, dev);
  cudaDeviceSynchronize();


  cudaEventRecord(start);
  for (int i = 0; i < 15; i++){
    gemmNaiveKernel<<<gridDim, blockDim>>>(um_A, um_B, um_C, M, N, K, layout_A, layout_B);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms_pf = 0.0f;
  cudaEventElapsedTime(&ms_um, start, stop);

  ms_pf /= 100.0f;

  std::cout << "UM_Kernel time(PREFETCH) : " << ms_pf << " ms\n";

  // prefetch back to cpu before reading
  cudaMemPrefetchAsync(um_C, sizeC, cudaCpuDeviceId);
  cudaDeviceSynchronize();

  #endif

  std::vector<float> h_C_um(M * N);
  std::memcpy(h_C_um.data(), um_C, sizeC);

  cudaFree(um_A);
  cudaFree(um_B);
  cudaFree(um_C);

  std::cout << "\n[UM] Comparing Unified Memory result vs reference:\n";
  compare_C(h_C_um, h_C_ref, M, N);

  #endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
