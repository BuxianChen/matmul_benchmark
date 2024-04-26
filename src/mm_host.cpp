template<typename T>
void mm_host(T const *A, T const *B, T *C, int m, int n, int k){
    // T 是 basetype, const 与 * 都是修饰符 modifier
    // T const *A 表示指针指向的值不能修改, 而 T* const A 表示 A 本身不能修改
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < k; ++col)
            for (int i = 0; i < n; i++)
                C[row*k+col] += A[row*n+i] * B[i*k+col];
                // accumulate_value(
                //     C, m, k, row, col,
                //     get_value(A, m, n, row, i) * get_value(B, n, k, i, col);
                // )
}

// 模板特化, 模板函数一般要写在头文件中, 而不是 cpp 文件中
template void mm_host<float>(float const *A, float const *B, float *C, int m, int n, int k);
