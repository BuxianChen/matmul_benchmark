# 杂记

[https://github.com/leimao/CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization)

`./build/src/profile_cuda_gemm_fp32`

```
Device Name: NVIDIA GeForce GTX 1650
Memory Size: 3.99969 GB
Peak Bandwitdh: 192.032 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V00
cuBLAS GEMM Kernel Performance
Latency: 74.2369 ms
Effective Bandwidth: 2.71195 GB/s
Effective TFLOPS: 1.85136 TFLOPS
Custom GEMM Kernel Performance
Latency: 5357.35 ms
Effective Bandwidth: 0.0375795 GB/s
Effective TFLOPS: 0.0256543 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.3857%

Custom GEMM Kernel V01
cuBLAS GEMM Kernel Performance
Latency: 72.929 ms
Effective Bandwidth: 2.76058 GB/s
Effective TFLOPS: 1.88456 TFLOPS
Custom GEMM Kernel Performance
Latency: 439.681 ms
Effective Bandwidth: 0.457892 GB/s
Effective TFLOPS: 0.312588 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 16.5868%

Custom GEMM Kernel V02
cuBLAS GEMM Kernel Performance
Latency: 70.6648 ms
Effective Bandwidth: 2.84904 GB/s
Effective TFLOPS: 1.94494 TFLOPS
Custom GEMM Kernel Performance
Latency: 324.131 ms
Effective Bandwidth: 0.621127 GB/s
Effective TFLOPS: 0.424023 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 21.8013%

Custom GEMM Kernel V02 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 71.0933 ms
Effective Bandwidth: 2.83187 GB/s
Effective TFLOPS: 1.93322 TFLOPS
Custom GEMM Kernel Performance
Latency: 479.966 ms
Effective Bandwidth: 0.41946 GB/s
Effective TFLOPS: 0.286352 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 14.8122%

Custom GEMM Kernel V03
cuBLAS GEMM Kernel Performance
Latency: 72.706 ms
Effective Bandwidth: 2.76905 GB/s
Effective TFLOPS: 1.89034 TFLOPS
Custom GEMM Kernel Performance
Latency: 149.717 ms
Effective Bandwidth: 1.34471 GB/s
Effective TFLOPS: 0.917989 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 48.5621%

Custom GEMM Kernel V03 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 70.7154 ms
Effective Bandwidth: 2.847 GB/s
Effective TFLOPS: 1.94355 TFLOPS
Custom GEMM Kernel Performance
Latency: 248.584 ms
Effective Bandwidth: 0.809895 GB/s
Effective TFLOPS: 0.552888 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 28.4473%

Custom GEMM Kernel V04
cuBLAS GEMM Kernel Performance
Latency: 70.9612 ms
Effective Bandwidth: 2.83714 GB/s
Effective TFLOPS: 1.93682 TFLOPS
Custom GEMM Kernel Performance
Latency: 66.9787 ms
Effective Bandwidth: 3.00583 GB/s
Effective TFLOPS: 2.05198 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 105.946%

Custom GEMM Kernel V04 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 71.1566 ms
Effective Bandwidth: 2.82934 GB/s
Effective TFLOPS: 1.9315 TFLOPS
Custom GEMM Kernel Performance
Latency: 65.6015 ms
Effective Bandwidth: 3.06893 GB/s
Effective TFLOPS: 2.09506 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 108.468%

Custom GEMM Kernel V05
cuBLAS GEMM Kernel Performance
Latency: 70.7312 ms
Effective Bandwidth: 2.84636 GB/s
Effective TFLOPS: 1.94312 TFLOPS
Custom GEMM Kernel Performance
Latency: 87.3322 ms
Effective Bandwidth: 2.3053 GB/s
Effective TFLOPS: 1.57375 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 80.9909%

Custom GEMM Kernel V05 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 70.6696 ms
Effective Bandwidth: 2.84884 GB/s
Effective TFLOPS: 1.94481 TFLOPS
Custom GEMM Kernel Performance
Latency: 68.9116 ms
Effective Bandwidth: 2.92152 GB/s
Effective TFLOPS: 1.99442 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 102.551%

Custom GEMM Kernel V06
cuBLAS GEMM Kernel Performance
Latency: 71.3226 ms
Effective Bandwidth: 2.82276 GB/s
Effective TFLOPS: 1.927 TFLOPS
Custom GEMM Kernel Performance
Latency: 74.7017 ms
Effective Bandwidth: 2.69508 GB/s
Effective TFLOPS: 1.83984 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 95.4766%

Custom GEMM Kernel V06 Vectorized
C[36, 1057] = 25893 C_ref[36, 1057] = 25896 Abs Diff: 3 Abs Diff Threshold: 0.001 Rel->Abs Diff Threshold: 0
profile_cuda_gemm_fp32: /mnt/include/profile_utils.cuh:352: std::pair<float, float> profile_gemm(size_t, size_t, size_t, size_t, size_t, size_t, std::function<void(long unsigned int, long unsigned int, long unsigned int, const T*, const T*, long unsigned int, const T*, long unsigned int, const T*, T*, long unsigned int, CUstream_st*)>, T, double, size_t, size_t, unsigned int) [with T = float; typename std::enable_if<((std::is_same<T, float>::value || std::is_same<T, double>::value) || std::is_same<T, __half>::value), bool>::type <anonymous> = true; size_t = long unsigned int]: Assertion `all_close<T>(C_host_from_device, C_host_ref, m, n, ldc, abs_tol, rel_tol)' failed.
Aborted
```

**关于 cublas_v2.h 和 cublas.h 的猜测**

```
header/
  - mylib.h
  - mylib_v2.h 
mylib.cpp
mylib_v2.cpp
main.cpp
```

文件内容

`mylib.h`

```cpp
// mylib.h
#ifndef MYLIB_H
#define MYLIB_H
void foo(int x);
#endif // MYLIB_H
```

`mylib_v2.h`

```cpp
// mylib_v2.h
#ifndef MYLIB_V2_H
#define MYLIB_V2_H
void foo_v2(int x); // 内部使用的实际函数名不同
// 重定义foo函数，使得它实际上调用foo_v2
#define foo(x) foo_v2(x)
#endif // MYLIB_V2_H
```

`mylib.cpp`

```cpp
//mylib.cpp
#include "mylib.h"
#include <iostream>

void foo(int x) {
    std::cout << "This is V1" << std::endl;
}
```

`mylib_v2.cpp`

```c++
// mylib_v2.cpp
#include "mylib_v2.h"
#include <iostream>

void foo_v2(int x) {
    std::cout << "This is V2" << std::endl;
}
```

`main.cpp`

```c++
// main.cpp
#include "mylib.h"
// #include "mylib_v2.h"

int main() {
    foo(1); // 这将调用 foo 或 foo_v2，取决于包含了哪个头文件, 但是如果同时包含 v1 和 v2, 也没有报编译错误, 而是指向 v2
    return 0;
}
```

编译指令

```bash
g++ -c -fPIC mylib.cpp -o mylib.o -Iheader
g++ -c -fPIC mylib_v2.cpp -o mylib_v2.o -Iheader
g++ -shared -o libmylib.so mylib.o mylib_v2.o
g++ main.cpp -o main -L. -lmylib -Iheader
./main
```

**关于 cublas 的 GEMM 相关的 API**

目标是计算 (注意调用者需保证 A, B, C 矩阵都是按 col major 进行存储, 具体细节下面链接里的接口文档说明的很清楚):

$$
C=\alpha \text{op}(A) \text{op}(B) + \beta C \\ \text{op}(A): (m, k), \\ \text{op}(B): (k, n) \\ C: (m, n)
$$

其中:

- `transa=CUBLAS_OP_N`: `op(A)=A`
- `transa=CUBLAS_OP_T`: `op(A)=A^T`, 即转置
- `transa=CUBLAS_OP_C`: `op(A)=A^H`, 即共轭转置


`cublas_v2.h:cublasSgemm`: [https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm), `cublasSgemm` 是单精度 (float32, fp32) 矩阵乘法, `cublasDgemm` 是双精度 (float64), `cublasCgemm` 是单精度复数, `cublasZgemm` 是双精度复数, `cublasHgemm` 是半精度 (float16, fp16).

`cublas_v2.h:cublasSgemmEx`: [https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmex](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmex), `cublasCgemmEx` 是复数矩阵乘法. 注意运算过程使用 fp32 进行计算, 但入参和出参矩阵的精度可以低于 fp32. 也就是支持入参的存储类型选择, 但运算时所用数据类型固定为 fp32.

`cublas_v2.h:cublasGemmEx`: [https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex), 支持出入参的存储类型选择, 运算时所用数据类型选择, 算法选择.

下面接口里 `lda` 表示 leading dimension, 当 `transa=CUBLAS_OP_N` 时, `lda>=max(1,m)`, 且分配的空间必须至少为 `(lda>=m, k)`, 否则 `lda>=max(1, k)`, 且分配的空间至少为 `(lda>=k, m)`.

可以参考这个 [youtube 视频](https://www.youtube.com/watch?v=PhjildK5oO8). lda 应该这么理解, 当矩阵 A 以列优先的方式进行存储的时候 (也就是先存储第一列, 再存储第二列, ...),

```c++
// M=3 是矩阵 A 的行数, N=4 是矩阵 A 的列数
/*
A = (
    a11, a12, a13, a14,
    a21, a22, a23, a24,
    a31, a32, a33, a34
)
*/
float *p = new float[M*N]  // 我们这里按列存储的方式存, 即 p = [a11, a21, a31, a12, a22, a32, a13, a23, a33, a14, a24, a34]
float *q = p + 4; // 我们将起始地址移到 a22 的未知

// 我们假设要知道以 q 作为左上角的子矩阵的 第 i 行与第 j 列的元素的值, 假设子矩阵的维度是 m * n, 应该这么计算:
q[j * M + i]  // 对于子矩阵 q 来说, M 就是所谓的 lda
```

也就是说, 在列优先存储的情况下: lda 指的是, `A` 矩阵可能是某一个更大的矩阵的子矩阵, 而这个更大的子矩阵的行数就是 lda.


```c++
// 函数声明 (函数原型)

// op(A) 矩阵是 m x k 的, 但是按列优先存储, op(B) 是 k * n 的, C 是 m * n 的
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,  // 这里传的是指针
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *beta,  // 这里传的是指针
    float *C,
    int ldc
)

cublasStatus_t cublasSgemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float    *alpha,
    const void     *A,
    cudaDataType_t Atype,
    int lda,
    const void     *B,
    cudaDataType_t Btype,
    int ldb,
    const float    *beta,
    void           *C,
    cudaDataType_t Ctype,
    int ldc)

cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void    *alpha,
    const void     *A,
    cudaDataType_t Atype,
    int lda,
    const void     *B,
    cudaDataType_t Btype,
    int ldb,
    const void    *beta,
    void           *C,
    cudaDataType_t Ctype,
    int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
)
```

符合 `cublasSgemm` 接口形式的 CPU 代码可以这样实现: TODO `./useless/sgemm_cpu.cpp`

最后, 我们来看下按 C 的 row major 应该怎么传参?

```c++
// A, B, C: row major, C = \alpha*A@B + \beta*C
// A: (m, k), B: (k, n), C(m, n), lda>=k, ldb>=n, ldc>=n
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc)
```

原因: TODO
