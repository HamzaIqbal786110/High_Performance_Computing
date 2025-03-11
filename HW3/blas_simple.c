#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

int main() 
{
    // Matrix dimensions
    int m = 2, n = 3, k = 2;
    
    // Allocate memory for matrices
    float *A = (float *)malloc(m * k * sizeof(float));
    float *B = (float *)malloc(k * n * sizeof(float));
    float *C = (float *)malloc(m * n * sizeof(float));
    
    // Initialize matrix values (example)
    for (int i = 0; i < m * k; i++) 
    {
        A[i] = i + 1;
    }
    
    for (int i = 0; i < k * n; i++) 
    {
        B[i] = i + 1;
    }
    
    // Perform matrix multiplication using cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, k, B,
    n, 0.0f, C, n);
    
    // Print the result matrix C
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            printf("%.2f ", C[i * n + j]);
        }
    printf("\n");
    }
    
    // Free allocated memory
    free(A);
    free(B);
    free(C);
    return 0;
}
