// nvcc useless/sgemm_gpu.cu -lcublas
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main(){
    /*
    A: (m=2, k=3)
    [
        0, 1, 2
        3, 4, 5
    ]
    -> [0, 3, 1, 4, 2, 5]

    B: (n=4, k=3)
    [
        0, 1, 2
        3, 4, 5
        6, 7, 8
        9, 10, 11
    ]
    -> [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

    C = A@B^T: (m=2, n=4)
    [
        5, 14, 23, 32
        14, 50, 86, 122
    ]
    -> [5, 14, 14, 50, 23, 86, 32, 122]
    */

    int m = 2, n = 4, k = 3;
    int lda = 2, ldb = 4, ldc = 2;
    const float alpha=1.0;
    const float beta=0.0;

    float *A = new float[m*k];
    float *B = new float[n*k];
    float *C = new float[m*n];
    A[0]=0; A[1]=3; A[2]=1; A[3]=4; A[4]=2; A[5]=5;
    B[0]=0; B[1]=3; B[2]=6; B[3]=9; B[4]=1; B[5]=4;
    B[6]=7; B[7]=10; B[8]=2; B[9]=5; B[10]=8; B[11]=11;
    C[0]=C[1]=C[2]=C[3]=C[4]=C[5]=C[6]=C[7]=0.0;

    float* A_device{nullptr};
    float* B_device{nullptr};
    float* C_device{nullptr};

    cudaMalloc(&A_device, m * k * sizeof(float));
    cudaMalloc(&B_device, n * k * sizeof(float));
    cudaMalloc(&C_device, m * n * sizeof(float));

    cudaMemcpy(A_device, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_device, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        m, n, k,
        &alpha, A_device, lda, B_device, ldb,
        &beta, C_device, ldc
    );

    cudaStreamSynchronize(stream);
    cudaMemcpy(C, C_device, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m * n; i++)
        std::cout << C[i] << ", ";

    return 0;
}