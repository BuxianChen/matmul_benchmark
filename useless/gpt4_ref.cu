// nvcc useless/gpt4_ref.cu -lcublas -lcurand -o bin/release/gpt4_ref 
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Error checking utilities
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Random initialization utility
// void randomInit(float* data, int size) {
//     curandGenerator_t gen;
//     curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//     curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
//     curandGenerateUniform(gen, data, size);
//     curandDestroyGenerator(gen);
// }

// Random initialization utility
void randomInit(float* data, int size){
    // 创建随机数引擎
    std::random_device rd;
    std::mt19937 gen(rd()); // 使用 Mersenne Twister 算法
    // 创建-1到1之间的均匀分布
    std::uniform_real_distribution<> distribution(-1.0, 1.0);
    for (int i = 0; i < size; i++)
        data[i] = distribution(gen);
}

// CPU matrix multiplication
void cpuMatrixMultiply(float* A, float* B, float* C, int N, int M, int K, float alpha, float beta) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < M; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * M + col];
            }
            C[row * M + col] = alpha * sum + beta * C[row * M + col];
        }
    }
}

// Function to compare results with a tolerance
bool validateResults(float* A, float* B, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (std::fabs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions
    int N = 254;
    int M = 255;
    int K = 256;
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    float alpha = 1.0f;
    float beta = 0.0f;
    const float tolerance = 0.01f;

    // Allocate host memory
    A = (float *)malloc(N * K * sizeof(float));
    B = (float *)malloc(K * M * sizeof(float));
    C = (float *)malloc(N * M * sizeof(float));

    // Allocate device memory
    gpuErrchk(cudaMalloc((void **)&d_A, N * K * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_B, K * M * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_C, N * M * sizeof(float)));

    // Initialize the input data
    randomInit(A, N * K);
    randomInit(B, K * M);
    randomInit(C, N * M);

    // Copy matrices from the host to the device
    gpuErrchk(cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_C, C, N * M * sizeof(float), cudaMemcpyHostToDevice));

    // Create a handle for cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform the matrix multiplication on the GPU
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_B, M, d_A, K, &beta, d_C, M);

    // Copy the result back to host
    gpuErrchk(cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    // Allocate memory for CPU result and copy the original C matrix
    float* C_cpu = (float *)malloc(N * M * sizeof(float));
    memcpy(C_cpu, C, N * M * sizeof(float));

    // Perform the matrix multiplication on the CPU for validation
    cpuMatrixMultiply(A, B, C_cpu, N, M, K, alpha, beta);

    // Validate the results
    if (validateResults(C, C_cpu, N * M, tolerance)) {
        std::cout << "Results are correct within tolerance." << std::endl;
    } else {
        std::cerr << "Results exceed tolerance." << std::endl;
    }

    // Clean up memory
    free(A);
    free(B);
    free(C);
    free(C_cpu);
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));

    // Destroy the cuBLAS handle
    cublasDestroy(handle);

    return 0;
}
