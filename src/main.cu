#include "mm_impl.cuh"
#include <cstdlib>
#include <cstdio>

int main(int argc, char **argv){
    size_t m=2, n=4, k=3;
    m = static_cast<size_t>(parse_cmdline_int(argc, argv, "-m", m));
    n = static_cast<size_t>(parse_cmdline_int(argc, argv, "-n", n));
    k = static_cast<size_t>(parse_cmdline_int(argc, argv, "-k", k));

    size_t lda = k, ldb = n, ldc = n;

    float *A = (float *)malloc(m * lda * sizeof(float));
    float *B = (float *)malloc(k * ldb * sizeof(float));
    float *C_cpu_ref = (float *)malloc(m * ldc * sizeof(float));
    float *C = (float *)malloc(m * ldc * sizeof(float));

    float const alpha = 1.0; // TODO: 为什么一定要加 const
    float const beta = 0.0;

    seq_init(A, m, k);
    seq_init(B, k, n);
    const_init(C_cpu_ref, m, n, 0.0);
    const_init(C, m, n, 0.0);

    float *A_device{nullptr};
    float *B_device{nullptr};
    float *C_device{nullptr};

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&A_device, m * lda * sizeof(float));
    cudaMalloc(&B_device, k * ldb * sizeof(float));
    cudaMalloc(&C_device, m * ldc * sizeof(float));

    cudaMemcpy(A_device, A, m * lda * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, k * ldb * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_device, C, m * ldc * sizeof(float), cudaMemcpyHostToDevice);

    launch_gemm_kernel_v01(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpy(C, C_device, m * ldc * sizeof(float), cudaMemcpyDeviceToHost);

    mm_host(m, n, k, &alpha, A, lda, B, ldb, &beta, C_cpu_ref, ldc);

    printf("====matrix A:====\n");
    print_matrix(A, m, k);
    printf("====matrix B:====\n");
    print_matrix(B, k, n);
    printf("====matrix C_cpu_ref:====\n");
    print_matrix(C_cpu_ref, m, n);
    printf("====matrix C:====\n");
    print_matrix(C, m, n);

    CHECK_CUDA_ERROR(cudaFree(A_device));
    CHECK_CUDA_ERROR(cudaFree(B_device));
    CHECK_CUDA_ERROR(cudaFree(C_device));
    // CHECK_CUDA_ERROR(cudaFreeHost(A));
    // CHECK_CUDA_ERROR(cudaFreeHost(B));
    // CHECK_CUDA_ERROR(cudaFreeHost(C));
    // CHECK_CUDA_ERROR(cudaFreeHost(C_cpu_ref));
    free(A);
    free(B);
    free(C);
    free(C_cpu_ref);
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return 0;
}