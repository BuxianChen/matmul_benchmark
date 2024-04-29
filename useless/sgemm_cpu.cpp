// c++ useless/sgemm_cpu.cpp
#include <iostream>
#define CIDX2C(i,j,ld) (((j)*(ld))+(i))  // column major in 0-based index, ld is rows of the allocated matrix
#define RIDX2C(i,j,ld) (((i)*(ld))+(j))  // row major in 0-based index, ld is columns of the allocated matrix

// 函数原型的修改: 返回参数简单改为了 void, 不包含 handle 参数
void cublasSgemm(
    int transa,  // 0 -> N, 1 -> T, 2 -> C
    int transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const float *A,  // 按列优先存储
    int lda,
    const float *B,  // 按列优先存储
    int ldb,
    const float *beta,
    float *C,  // 按列优先存储
    int ldc
){
    int a, b;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            C[CIDX2C(i, j, ldc)] = (*beta) * C[CIDX2C(i, j, ldc)];
            for (int idx = 0; idx < k; ++idx){
                // if transa == 2, use conj(A[CIDX2C(i, idx, lda)])
                a = (transa == 0) ? CIDX2C(i, idx, lda): CIDX2C(idx, i, lda);
                b = (transb == 0) ? CIDX2C(idx, j, ldb): CIDX2C(j, idx, ldb);
                C[CIDX2C(i, j, ldc)] += (*alpha) * A[a] * B[b];
            }
        }
}

int col_major_test() {
    int A_row = 8, A_col = 16;
    int B_row = 10, B_col = 20;
    int C_row = 12, C_col = 14;
    
    int m = 2, n = 4, k = 3;
    float *A_alloc = new float[A_row * A_col];
    float *B_alloc = new float[B_row * B_col];
    float *C_alloc = new float[C_row * C_col];

    int transa = 0;
    int transb = 1;  // only need change this line to switch test case.

    // ld{x} alway equals to allocated number of rows
    int lda = A_row;
    int ldb = B_row;
    int ldc = C_row;

    int A_start_i = 1, A_start_j = 2;
    int B_start_i = 3, B_start_j = 6;
    int C_start_i = 5, C_start_j = 7;

    float *A = A_alloc + A_start_j * A_row + A_start_i;
    float *B = B_alloc + B_start_j * B_row + B_start_i;
    float *C = C_alloc + C_start_j * C_row + C_start_i;

    for (int i = 0; i < A_row; i++)
        for (int j = 0; j < A_col; j++)
            A_alloc[CIDX2C(i, j, A_row)] = RIDX2C(i, j, A_col);

    for (int i = 0; i < B_row; i++)
        for (int j = 0; j < B_col; j++)
            B_alloc[CIDX2C(i, j, B_row)] = RIDX2C(i, j, B_col);
    
    for (int i = 0; i < C_row; i++)
        for (int j = 0; j < C_col; j++)
            C_alloc[CIDX2C(i, j, C_row)] = 0.0;

    for (int i = 0; i < 10; i++)
        std::cout << i << " A: " << A_alloc[i] << ", B: " << B_alloc[i] << ", C: " << C_alloc[i] << std::endl;

    /*
        A1, A2: (2, 3)
        [
            2+16*1, 3+16*1, 4+16*1,
            2+16*2, 3+16*2, 4+16*2,
        ]

        B1: (4, 3), transb = 1
        [
            6+20*3, 7+20*3, 8+20*3,
            6+20*4, 7+20*4, 8+20*4,
            6+20*5, 7+20*5, 8+20*5,
            6+20*6, 7+20*6, 8+20*6,
        ]

        C1=A1@B1^T: (2, 4)
        [
            3821,  4961,  6101,  7241,
            7037,  9137, 11237, 13337
        ]

        B2: (3, 4) transb = 0
        [
            6+20*3, 7+20*3, 8+20*3, 9+20*3,
            6+20*4, 7+20*4, 8+20*4, 9+20*4,
            6+20*5, 7+20*5, 8+20*5, 9+20*5
        ]

        C2=A2@B2: (2, 4)
        [
            4942, 4999, 5056, 5113,
            9070, 9175, 9280, 9385
        ]
    */

    float alpha_float = 1.0;
    float beta_float = 0.0;
    float *alpha = &alpha_float;
    float *beta = &beta_float;

    cublasSgemm(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc
    );


    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            std::cout << C[CIDX2C(i, j, ldc)] << ", ";
        std::cout << std::endl;
    }

    delete[] A_alloc;
    delete[] B_alloc;
    delete[] C_alloc;

    return 0;
}



int row_major_test() {
    int A_row = 8, A_col = 16;
    int B_row = 10, B_col = 20;
    int C_row = 12, C_col = 14;
    
    int m = 2, n = 4, k = 3;
    float *A_alloc = new float[A_row * A_col];
    float *B_alloc = new float[B_row * B_col];
    float *C_alloc = new float[C_row * C_col];

    int transa = 0;
    int transb = 1;

    // things changed: ld{x} alway equals to allocated number of columns (for row major)
    int lda = A_col;
    int ldb = B_col;
    int ldc = C_col;

    int A_start_i = 1, A_start_j = 2;
    int B_start_i = 3, B_start_j = 6;
    int C_start_i = 5, C_start_j = 7;

    float *A = A_alloc + A_start_i * lda + A_start_j;
    float *B = B_alloc + B_start_i * ldb + B_start_j;
    float *C = C_alloc + C_start_i * ldc + C_start_j;

    for (int i = 0; i < A_row; i++)
        for (int j = 0; j < A_col; j++)
            A_alloc[RIDX2C(i, j, A_col)] = RIDX2C(i, j, A_col);

    for (int i = 0; i < B_row; i++)
        for (int j = 0; j < B_col; j++)
            B_alloc[RIDX2C(i, j, B_col)] = RIDX2C(i, j, B_col);
    
    for (int i = 0; i < C_row; i++)
        for (int j = 0; j < C_col; j++)
            C_alloc[RIDX2C(i, j, C_col)] = 0.0;

    for (int i = 0; i < 20; i++)
        std::cout << i << " A: " << A_alloc[i] << ", B: " << B_alloc[i] << ", C: " << C_alloc[i] << std::endl;

    /*
        A1, A2: (2, 3)
        [
            2+16*1, 3+16*1, 4+16*1,
            2+16*2, 3+16*2, 4+16*2,
        ]

        B1: (4, 3), transa = 1
        [
            6+20*3, 7+20*3, 8+20*3,
            6+20*4, 7+20*4, 8+20*4,
            6+20*5, 7+20*5, 8+20*5,
            6+20*6, 7+20*6, 8+20*6,
        ]

        C1=A1@B1^T: (2, 4)
        [
            3821,  4961,  6101,  7241,
            7037,  9137, 11237, 13337
        ]

        B2: (3, 4) transa = 0
        [
            6+20*3, 7+20*3, 8+20*3, 9+20*3,
            6+20*4, 7+20*4, 8+20*4, 9+20*4,
            6+20*5, 7+20*5, 8+20*5, 9+20*5
        ]

        C2=A2@B2: (2, 4)
        [
            4942, 4999, 5056, 5113,
            9070, 9175, 9280, 9385
        ]
    */

    float alpha_float = 1.0;
    float beta_float = 0.0;
    float *alpha = &alpha_float;
    float *beta = &beta_float;

    cublasSgemm(
        transb,  // things changed: switch `transa, transb` -> `transn, transa`
        transa,
        n,       // things changed: switch `m, n, k` -> `n, m, k`
        m,
        k,
        alpha,
        B,       // things changed: switch `A, lda, B, ldb` -> `B, ldb, A, lda`
        ldb,
        A,
        lda,
        beta,
        C,
        ldc
    );


    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            std::cout << C[RIDX2C(i, j, ldc)] << ", ";
        std::cout << std::endl;
    }

    delete[] A_alloc;
    delete[] B_alloc;
    delete[] C_alloc;

    return 0;
}

int main(){
    std::cout << "column major test" << std::endl;
    col_major_test();
    std::cout << "row major test" << std::endl;
    row_major_test();
    return 0;
}