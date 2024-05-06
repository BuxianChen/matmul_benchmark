#include "mm_impl.cuh"

template <typename T>
__global__ void gemm_v01(
    size_t m,
    size_t n,
    size_t k,
    T alpha,
    T const *A,
    size_t lda,
    T const *B,
    size_t ldb,
    T beta,
    T *C,
    size_t ldc
){
    size_t col{blockIdx.x * blockDim.x + threadIdx.x}; // 连续变化
    size_t row{blockIdx.y * blockDim.y + threadIdx.y}; // 不变

    if (row < m && col < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t i = 0; i < k; ++i)
            // A 不变, B 连续变化
            sum += A[row * lda + i] * B[i * ldb + col];
        // C 连续变化
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

template <typename T>
void launch_gemm_kernel_v01(
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
    size_t ldc,
    cudaStream_t stream
){
    dim3 const block_dim{BLOCK_SIZE, BLOCK_SIZE, 1};
    // TODO: static_cast<unsigned int>(m) 避免警告?
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1) / block_dim.y,
    };
    gemm_v01<T><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
}

// Explicit instantiation.
template void launch_gemm_kernel_v01<float>(
    size_t m,
    size_t n,
    size_t k,
    float const *alpha,
    float const *A,
    size_t lda,
    float const *B,
    size_t ldb,
    float const *beta,
    float *C,
    size_t ldc,
    cudaStream_t stream
);