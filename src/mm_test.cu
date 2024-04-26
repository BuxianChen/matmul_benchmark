#include "mm_impl.hpp"
#include <cstdlib>
#include <cstdio>


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    float* elements;
    int height;
    int width;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}




int main(int argc, char **argv){
    int m=64, n=64, k=64;
    m = parse_cmdline_int(argc, argv, "-m", m);
    n = parse_cmdline_int(argc, argv, "-n", n);
    k = parse_cmdline_int(argc, argv, "-k", k);

    float *A = (float *)malloc(m * n * sizeof(float));
    float *B = (float *)malloc(n * k * sizeof(float));
    float *C = (float *)malloc(m * k * sizeof(float));
    float *D = (float *)malloc(m * k * sizeof(float));

    random_init(A, m, n);
    random_init(B, n, k);
    const_init(C, m, k, 0.0);
    const_init(D, m, k, 0.0);

    mm_host(A, B, C, m, n, k);
    MatMul(Matrix{A, m, n}, Matrix{B, n, k}, Matrix{D, m, k});

    printf("====matrix A:====\n");
    print_matrix(A, m, n);
    printf("====matrix B:====\n");
    print_matrix(B, n, k);
    printf("====matrix C:====\n");
    print_matrix(C, m, k);
    printf("====matrix D:====\n");
    print_matrix(D, m, k);

    return 0;
}