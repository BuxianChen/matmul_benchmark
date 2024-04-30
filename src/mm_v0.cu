#include "mm_impl.cuh"

template <typename T>
__global__ void gemm_v00(
    size_t m, size_t n, size_t k, T alpha,
    T const* A, size_t lda, T const *B, size_t ldb,
    T beta, T* C, size_t ldc
){
    size_t row{blockIdx.x * blockDim.x + threadIdx.x};  // threadIdx.x 变化最快, 因此 row 随着 thread 连续变化
    size_t col{blockIdx.y * blockDim.y + threadIdx.y};  // 相邻 thread 基本不变

    if (row < m && col < n){
        T sum{static_cast<T>(0)};
        for (size_t i = 0; i < k; ++i)
            sum += A[row * lda + i] * B[i * ldb + col];
            // A 的读取在相邻的 thread 间会跳跃 lda 个元素, 因此基本上每次迭代对于一个warp来说要做 warp_size=32 次 cache
            // 同一个 warp 中的 thread 会读取同一个 B 的元素: 第1次是读 B[0+col], 第2次是读 B[ldb+col], 因此每次迭代对于一个warp来说都要做 1 次 cache
            // A 的情况最为糟糕, B 会好些, 但更理想的是 B 可以做一个 cache 预先保存 {B[col], B[ldb+col], ... B[(cache_size-1)*ldb+col]}
            // 这样一来, 第 0-cache_size 个迭代中便可以共享这个 cache 池

        // C 的读取在相邻的 thread 间会跳跃 ldc 个元素, 所以对于一个 warp 来说, 需要做 warp_size=32 次 cache
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{BLOCK_SIZE, BLOCK_SIZE, 1};
    // TODO: static_cast<unsigned int>(m) 避免警告?
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1) / block_dim.y,
    };
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc
    );
}


// Explicit instantiation.
template void launch_gemm_kernel_v00<float>(size_t m, size_t n, size_t k,
                                            float const* alpha, float const* A,
                                            size_t lda, float const* B,
                                            size_t ldb, float const* beta,
                                            float* C, size_t ldc,
                                            cudaStream_t stream);