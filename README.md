# MatMul benchmark

矩阵乘法的 C++ 及 CUDA 实现

计划参考的内容:

- [https://docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
- [https://github.com/leimao/CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization)
- [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)

当前进度:

mm_host.cpp: c++ 单进程实现
mm_test.cu: 参考自 [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)

```
make build
./bin/release/main -m 16 -n 16 -k 16
make mm_test
./bin/release/mm_test -m 16 -n 16 -k 16
```

TODO:

- 把 Makefile 写好一些, 把 .vscode 文件夹精简下, 感觉应该更多依赖于 Makefile 或 CMakeList 而不是依赖 .vscode/tasks.json 和 launch.json
- 加上计时器
- 加上结果验证
- CUDA 代码加上更多的注释
- 删除无效/重复代码, 整理代码结构
