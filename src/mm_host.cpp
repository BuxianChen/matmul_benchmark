#include<cstdlib>

template<typename T>
void mm_host(size_t m, size_t n, size_t k,
    T const *alpha,
    T const *A, size_t lda,
    T const *B, size_t ldb,
    T const *beta,
    T *C, size_t ldc
){
    // T 是 basetype, const 与 * 都是修饰符 modifier
    // T const *A 表示指针指向的值不能修改, 而 T* const A 表示 A 本身不能修改
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col){
            T sum{static_cast<T>(0)};  // sum{0.0} 是 C++11 的 uniform initialization 语法
            for (int i = 0; i < k; i++)
                sum += A[row * lda + i] * B[i * ldb + col];
            C[row * ldc + col] = (*alpha) * sum + (*beta) * C[row * ldc + col];
        }
}

// 模板特化, 模板函数一般要写在头文件中, 而不是 cpp 文件中
template void mm_host<float>(size_t m, size_t n, size_t k,
    float const *alpha,
    float const *A, size_t lda,
    float const *B, size_t ldb,
    float const *beta,
    float *C, size_t ldc
);
