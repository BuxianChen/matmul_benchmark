// .h 是 C 的风格, .hpp 是 C++ 的风格, .cuh 是非官方约定的 cuda 头文件尾缀
#ifndef MM_IMPL_CUH
#define MM_IMPL_CUH
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

int parse_cmdline(int argc, char **argv, char const *option);
int parse_cmdline_int(int argc, char **argv, char const *option, int default_value);
void print_matrix(float *matrix, size_t row, size_t col);
void random_init(float *matrix, size_t row, size_t col);
void const_init(float *matrix, size_t row, size_t col, float value);
void seq_init(float *matrix, size_t row, size_t col);

template<typename T>
T get_value(T const *M, int row, int col, int i, int j);

template<typename T>
T set_value(T const *M, int row, int col, int i, int j, T value);

template<typename T>
T accumulate_value(T const *M, int row, int col, int i, int j, T value);


#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file,
                const int line);


template<typename T>
void mm_host(size_t m, size_t n, size_t k,
    T const *alpha,
    T const *A, size_t lda,
    T const *B, size_t ldb,
    T const *beta,
    T *C, size_t ldc
);

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v01(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

#endif