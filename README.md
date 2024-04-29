# MatMul benchmark

矩阵乘法的 C++ 及 CUDA 实现

计划参考的内容:

- [https://docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
- [https://github.com/leimao/CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization)
- [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)

当前进度:

mm_host.cpp: c++ 单进程实现

```
make build
./bin/release/main -m 16 -n 16 -k 16
```

## cublasSgemm equivalent implementation (finished)

- `sgemm_gpu.cu`: `cublasSgemm` 函数的使用样例
- `sgemm_cpu.cpp`: 与 `cublasSgemm` 接口完全对应的 C++ CPU 实现

## different matmul implementations (TODO)

以下实现与 `cublasSgemm` 的接口格式不一致, 并且不包含 `cublasSgemm` 提供的 `transa`, `transb` 参数, 在与 `cublasSgemm` 做对比时, 要注意 `cublasSgemm` 传递的实参.

```c++
template <typename T>
__global__ void gemm_vxx(size_t m, size_t n, size_t k, T const *alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T const *beta, T* C,
                         size_t ldc)
```


TODO:

- 把 Makefile 写好一些, 把 .vscode 文件夹精简下, 感觉应该更多依赖于 Makefile 或 CMakeList 而不是依赖 .vscode/tasks.json 和 launch.json
- 加上计时器
- 加上结果验证
- CUDA 代码加上更多的注释
- 删除无效/重复代码, 整理代码结构
