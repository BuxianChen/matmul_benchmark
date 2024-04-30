# 杂记

## CUDA-GEMM-Optimization 运行结果

参照 [https://github.com/leimao/CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization) 的 README 进行单精度性能测试:

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

## 关于 cublas_v2.h 和 cublas.h (new-and-legacy-cublas-api) 的猜测

官方描述: [https://docs.nvidia.com/cuda/cublas/index.html#new-and-legacy-cublas-api](https://docs.nvidia.com/cuda/cublas/index.html#new-and-legacy-cublas-api)

总结如下:

- 已经存在的上古代码里写的是 `#include <cublas.h>`, 这类代码的运行最终会指向 legacy API
- 新开发建议 `#include <cublas_v2.h>`, 这类代码的运行最终会指向 new API, 需要注意的是, legacy API 里已经定义的符号例如 `cublasSgemm`, 在 `cublas_v2.h` 中, 实际上是 `cublasSgemm_v2`, 但是使用了宏定义将 `cublasSgemm`, 因此可以直接使用 `cublasSgemm` 这个符号, 但是需要注意的是这些与 legacy API 同名的函数可能会有不同的形参
- 不要同时 include 两个头文件, 会引发编译错误. 备注: 实测下来发现如果 include 两个头文件, 把 `#include <cublas.h>` 放在 `#include <cublas_v2.h>` 之前, 有可能能通过编译及正常执行, 但是反过来则会报编译错误.

以下是验证上面所描述情形的例子:

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

## cublas GEMM 相关 API

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

---------

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

**layout**

总之, 明确一个子矩阵需要 4 个参数: 一个指针指向子矩阵的起始地址, 子矩阵的行数, 列数, ld.

在列优先的情况下:

```C++
// aij 表示逻辑上的第 i 行, 第 j 列的元素
aij = A[j*lda+i];  // lda = 分配空间时的矩阵的行数
```

在行优先的情况下:

```C++
// aij 表示逻辑上的第 i 行, 第 j 列的元素
aij = A[i*lda+j];  // lda = 分配空间时的矩阵的列数
```

----------------

cublas 关于矩阵乘法的函数原型如下:

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

最后, 我们来看下按 C 的 row major 应该怎么传参

```c++
// A, B, C: row major

// C = \alpha*A@B + \beta*C
// A: (m, k), B: (k, n), C(m, n)
lda = k; ldb = n; ldc = n;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)

// C = \alpha*A@B^T + \beta*C
lda = k, ldb = n, ldc = n;
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)
```

原因: TODO

因为行优先更符合 C++ 的习惯, 后续采用行优先的存储方案来实现矩阵乘法, 另外, 为降低实现难度, `cublasSgemm` 中的 `transa` 和 `transb` 之后都固定为 `CUBLAS_OP_N`. 因此用来作为参照组的 `cublasSgemm` 将会像上面的 row major 的方式进行调用.

## Coalesced Memory Access

Coalesced Memory Access 是对全局内存的优化考虑, 而 memory bank 是对共享内存的优化考量, 两者没有交集.

- 官方资料: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput)
- 博客: [https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/](https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/)

以下是从上面的博客中摘抄的所谓

> The global memory read is coalesced whereas the global memory write is not.

```c++
constexpr size_t div_up(size_t a, size_t b) { return (a + b - 1) / b; }

template <typename T>
__global__ void transpose_read_coalesced(T* output_matrix,
                                         T const* input_matrix, size_t M,
                                         size_t N)
{
    size_t const j{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const i{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const from_idx{i * N + j};
    if ((i < M) && (j < N))
    {
        size_t const to_idx{j * M + i};
        output_matrix[to_idx] = input_matrix[from_idx];
    }
}

template <typename T>
void launch_transpose_read_coalesced(T* output_matrix, T const* input_matrix,
                                     size_t M, size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(N, warp_size)),
                               static_cast<unsigned int>(div_up(M, warp_size))};
    transpose_read_coalesced<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output_matrix, input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}
```

这里仔细分析以下为什么是 read coalesced. 首先需要明确的是: `launch_transpose_read_coalesced` 的目标是将一个 row major, 且行数为 `M`, 列数为 `N` 的矩阵 `input_matrix` 转置, 注意 `output_matrix` 同样也是 row major 的, 也就是说期望的结果如下:

```c++
size_t M = 2, N = 3;
float *input_matrix;  // {2, 4, 6, 8, 9, 10} <-> [[2, 4, 6], [8, 9, 10]]
float *output_matrix;
launch_transpose_read_coalesced(output_matrix, input_matrix, M, N, stream);
// output_matrix: {2, 8, 4, 9, 6, 10} <-> [[2, 8], [4, 9], [6, 10]]
```

然后我们来看具体实现, 首先我们确认实现的正确性(此处从略): 注意 `block_per_grid.x = N / warp_size, block_per_grid.y = M / warp_size`, ...

最后我们来看其为何是 read coalesced. 注意到 kernel function 中:

```c++
size_t const j{threadIdx.x + blockIdx.x * blockDim.x};  // threadIdx.x 是变化最快的维度, 因此 j 是随着 thread 连续变化的
size_t const i{threadIdx.y + blockIdx.y * blockDim.y};  // 相邻 thread 的 i 基本上是相同的

size_t const from_idx{i * N + j};  // from_idx 是随着 thread 连续变化的
size_t const to_idx{j * M + i};    // to_idx 是随着 thread 跳跃变化的 (因为 j 是连续变化的, 而 j*M 造成跳跃变化)
output_matrix[to_idx] = input_matrix[from_idx];  // input_matrix 是 read, output_matrix 是 write
```

TODO: REMOVE THIS: 博客中实现矩阵转置的执行结果

不启用任何优化选项: `nvcc useless/transpose.cu -o transpose -Xptxas -O0 && ./transpose`

```
1280 x 1280 Matrix
Transpose Write Coalesced
Latency: 10.381 ms
Transpose Read Coalesced
Latency: 8.656 ms
Transpose Read and Write Coalesced
Latency: 8.336 ms
```

启用 `O3` 优化选项: `nvcc useless/transpose.cu -o transpose -Xptxas -O3 && ./transpose`

```
1280 x 1280 Matrix
Transpose Write Coalesced
Latency: 0.941 ms
Transpose Read Coalesced
Latency: 1.215 ms
Transpose Read and Write Coalesced
Latency: 0.514 ms
```

## Memory bank

Coalesced Memory Access 是对全局内存的优化考虑, 而 memory bank 是对共享内存的优化考量, 两者没有交集.

使用 shared memory 优于使用全局内存, 避免 bank conflict 优于有 bank conflict 的 shared memory 的使用

- 官方资料: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput)
- 官方资料: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x)
- 博客: [https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/)


以 V100 为例, shared memory per block 的大小为 49152 Byte, 这些字节被分为 32 组 (因此每组为 49152/32=1536 Byte), 每个组称为一个 bank, 每个 bank 在一个时钟周期内提供 32 bit (也就是 4 Byte) 个读写能力. 我们知道线程是以 32 个为一组同时执行的, 如果这 32 个线程分别访问 32 个 bank (在一个时钟周期内每个只能最多访问 4 Byte), 那么不会造成 bank conflict; 然而假设第一个线程访问 bank-1 的第 0-32 bit, 第二个线程访问 bank-1 的第 32-64 bit, 就会造成 bank conflict; 但是如果第一个线程和第二个线程访问的都是 bank-1 的 0-32 bit, 也不造成 bank conflict.

注意: bank 的排布方式是交错的, 即 

```c++
// 第 0 号 bank 的内存地址包括:
(0)~(32) bit, (1*32*32)~(1*32*32+32) bit, (2*32*32)~(2*32*32+32) bit, ..., (1535*32*32)~(1535*32*32+32) bit

// 第 1 号 bank 的内存地址包括:
(32)~(64) bit, (1*32*32+32)~(1*32*32+64) bit, (2*32*32+32)~(2*32*32+64) bit, ..., (1535*32*32+32)~(1535*32*32+64) bit
```

因此假设需要申请一块逻辑上是 2D 的 shared memory, 且同一个 block 且同一个 warp 中线程需要按行/列访问共享内存时, 推荐像这样申请一块 shared memory

```c++
BLOCK_SIZE = 32;
__shared__ float buffer[BLOCK_SIZE][BLOCK_SIZE + 1];
```

好处是:

- 假设 `block{i}-warp{j}-thread{0~31}` 需要分别访问 buffer 的一行 `buffer[k][0:32]`, 由于 buffer 是 row major 的, 且一个 float 刚好需要 32 bit, 因此 `buffer[k][0:32]` 分别属于不同的 bank, 这样同一个 warp 里的不同 thread 不会造成 banck confilct
- 假设 `block{i}-warp{j}-thread{0~31}` 需要分别访问 buffer 的一列 `buffer[0:32][k]`, 由于 buffer 是 row major 的, 且一个 float 刚好需要 32 bit, 由于 buffer 的列数为 33, 因此 `buffer[0:32][k]` 也分别属于不同的 bank, 这样同一个 warp 里的不同 thread 也不会造成 banck confilct

## A100 规格说明书

[https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21819-optimizing-applications-for-nvidia-ampere-gpu-architecture.pdf](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21819-optimizing-applications-for-nvidia-ampere-gpu-architecture.pdf), 比较有参考性, 也涉及了很多 GPU 的软硬件概念

## 性能分析的提示

在分析访存方面的性能分析时, 需要注意相邻的 thread 是属于同一个 warp 的, 它们每时每刻的指令都是相同的, warp 里的数据访问性能是关键.

缓存行大小 (Cache Line Size) 一般都是 64 Byte / 128 Byte. 注意与 L1 Cache Size 和 L2 Cache Size 做区分, L1 Cache Size 一般可以做到几十 KB, L2 Cache Size 一般是 1M 左右 (A100 是 40M)
