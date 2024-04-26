// .h 是 C 的风格, .hpp 是 C++ 的风格, .cuh 是非官方约定的 cuda 头文件尾缀
#ifndef MM_IMPL_HPP
#define MM_IMPL_HPP

int parse_cmdline(int argc, char **argv, char const *option);
int parse_cmdline_int(int argc, char **argv, char const *option, int default_value);
void print_matrix(float *matrix, int row, int col);
void random_init(float *matrix, int row, int col);
void const_init(float *matrix, int row, int col, float value);

template<typename T>
T get_value(T const *M, int row, int col, int i, int j);

template<typename T>
T set_value(T const *M, int row, int col, int i, int j, T value);

template<typename T>
T accumulate_value(T const *M, int row, int col, int i, int j, T value);

template<typename T>
void mm_host(T const *A, T const *B, T *C, int m, int n, int k);
#endif