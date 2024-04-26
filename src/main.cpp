#include "mm_impl.hpp"
#include <cstdlib>
#include <cstdio>

int main(int argc, char **argv){
    int m=64, n=64, k=64;
    m = parse_cmdline_int(argc, argv, "-m", m);
    n = parse_cmdline_int(argc, argv, "-n", n);
    k = parse_cmdline_int(argc, argv, "-k", k);

    float *A = (float *)malloc(m * n * sizeof(float));
    float *B = (float *)malloc(n * k * sizeof(float));
    float *C = (float *)malloc(m * k * sizeof(float));

    const_init(A, m, n, 2.0);
    const_init(B, n, k, 2.0);
    const_init(C, m, k, 0.0);
    mm_host(A, B, C, m, n, k);

    printf("====matrix A:====\n");
    print_matrix(A, m, n);
    printf("====matrix B:====\n");
    print_matrix(B, n, k);
    printf("====matrix C:====\n");
    print_matrix(C, m, k);

    return 0;
}