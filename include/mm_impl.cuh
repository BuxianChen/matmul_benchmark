// .h 是 C 的风格, .hpp 是 C++ 的风格, .cuh 是非官方约定的 cuda 头文件尾缀
#ifndef MM_IMPL_CUH
#define MM_IMPL_CUH
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include <cublas_v2.h>

#define BLOCK_SIZE 16


/*
用法: CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m*lda*sizeof(T)));
被宏替换为:
check_cuda(
    cudaMallocHost(&A_host, m*lda*sizeof(T)),
    "cudaMallocHost(&A_host, m*lda*sizeof(T))",
    "xxx.cu",
    123
);
*/
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(
    cudaError_t err,
    const char* const func,
    const char* const file,
    const int line
);

/*
用法: CHECK_LAST_CUDA_ERROR()
被宏替换为
check_cuda_last(
    "xxx.cu",
    123
)
*/
#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
void check_cuda_last(const char* const file, const int line);

#define CHECK_CUBLASS_ERROR(val) check_cublass((val), #val, __FILE__, __LINE__)
void check_cublass(
    cublasStatus_t err,
    const char* const func,
    const char* const file,
    const int line
);


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





template<typename T>
bool matrix_all_close(
    T const *output,
    T const *ref,
    size_t m,
    size_t n,
    size_t ld,
    T abs_tol,  // 转化为 double 计算
    double rel_tol
){
    bool status{true};
    for (size_t i{0U}; i < m; ++i){
        for (size_t j{0U}; j < n; j++){
            double const val{static_cast<double>output[i*ld+j]};
            double const ref_val{static_cast<double>ref[i*ld+j]};
            double const diff_val{std::abs(val-ref_val)};
            if (
                diff_val > std::max(
                    static_cast<double>(abs_tol),
                    static_cast<double>(std::abs(ref_val) * rel_tol)
                )
            ){
                std::cout << "output[" << i << ", " << j << "] = " << val <<
                    << " ref[" << i << ", " << j << "] = " << ref_val
                    << " Abs Diff: " << diff_val
                    << " Abs Diff Threshold: " << static_cast<double>(abs_tol)
                    << " Rel Diff Threshold: " << static_cast<double>(rel_tol)
                    << " Rel -> Abs Diff Threshold: "
                    << static_cast<double>(static_cast<double>(std::abs(ref_val) * rel_tol))
                    << std::endl;
                status = false;
                return status;
            }
        }
    }
    return status;
}


template<
    typename T,
    typename std::enable_if<
        std::is_same<T, float>::value ||
            std::is_same<T, double>::value,
        bool
    >::type = true
>
float profile_gemm(
    size_t m,
    size_t n,
    size_t k,
    T const *alpha,
    T const *A_host,
    size_t lda,
    T const *B_host,
    size_t ldb,
    T const *beta,
    T *C_host,
    size_t ldc,
    std::function<
        void(
            size_t m,
            size_t n,
            size_t k,
            T const* alpha,
            T const* A_device,
            size_t lda,
            T const* B_device,
            size_t ldb,
            T const* beta,
            T* C_device,
            size_t ldc,
            cudaStream_t stream
        )
    > gemm_kernel_launch_function,
    T *C_host_ref,
    T abs_tol,
    double rel_tol,
    size_t num_repeats = 10,
    size_t num_warmups = 10
){
    T* A_device{nullptr};
    T* B_device{nullptr};
    T* C_device{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m*lda*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device, m*ldb*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device, m*ldc*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_from_device, m*ldc*sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m*lda*sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_device, B_host, k*ldb*sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m*ldc*sizeof(T), cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    gemm_kernel_launch_function(
        m,
        n,
        k,
        alpha,
        A_device,
        lda,
        B_device,
        ldb,
        beta,
        C_device,
        ldc,
        stream
    );

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(
        cudaMemcpy(
            C_host_from_device,
            C_device,
            m * ldc * sizeof(T),
            cudaMemcpyDeviceToHost
        )
    );
    assert(
        matrix_all_close<T>(
            C_host_from_device,
            C_host_ref,
            m,
            n,
            ldc,
            abs_tol,
            rel_tol
        )
    );

    // TODO
}




template<typename T>
void mm_host(
    size_t m,
    size_t n,
    size_t k,
    T const *alpha,
    T const *A,
    size_t lda,
    T const *B,
    size_t ldb,
    T const *beta,
    T *C,
    size_t ldc
);

template <typename T>
void launch_gemm_kernel_v00(
    size_t m,
    size_t n,
    size_t k,
    T const* alpha,
    T const* A,
    size_t lda,
    T const* B,
    size_t ldb,
    T const* beta,
    T* C,
    size_t ldc,
    cudaStream_t stream
);

template <typename T>
void launch_gemm_kernel_v01(
    size_t m,
    size_t n,
    size_t k,
    T const* alpha,
    T const* A,
    size_t lda,
    T const* B,
    size_t ldb,
    T const* beta,
    T* C,
    size_t ldc,
    cudaStream_t stream
);

#endif