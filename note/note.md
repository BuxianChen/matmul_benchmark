
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
