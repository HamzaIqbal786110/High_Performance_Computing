#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#define N 256
#define LOOPS 10

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}


int main()
{
    float *a = (float*)malloc(N * N * sizeof(float)); /* input matrix */
    float *b = (float*)malloc(N * N * sizeof(float)); /* input matrix */
    float *c = (float*)malloc(N * N * sizeof(float)); /* result matrix */
    int i,j,k,l, num_zeros;
    double start, finish, total;
    
    /* initialize a dense matrix */
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            a[i * N + j] = (float)(i+j);
            b[i * N + j] = (float)(i-j);
        }
    }
    
    printf("starting dense matrix multiply \n");
    
    start = CLOCK();
    for(int i = 0; i < LOOPS; i++)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, a, N, b,
        N, 0.0f, c, N);
    }
    finish = CLOCK();
    total = finish-start;
    printf("a result %g \n", c[7 * N + 8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with dense matrices = %f ms\n",
    total);
   
    /* initialize a sparse matrix */
    num_zeros = 0;
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            if ((i<j)&&(i%2>0))
            {
                a[i * N + j] = (double)(i+j);
                b[i * N + j] = (double)(i-j);
            }
            else
            {
                num_zeros++;
                a[i * N + j] = 0.0;
                b[i * N + j] = 0.0;
            }
        }
    }

    printf("starting sparse matrix multiply \n");
    
    start = CLOCK();
    for (l=0; l<LOOPS; l++) 
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, a, N, b,
        N, 0.0f, c, N);
    }
    finish = CLOCK();
    
    total = finish-start;
    printf("A result %g \n", c[7 * N + 8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with sparse matrices = %f ms\n",
    total);
    printf("The sparsity of the a and b matrices = %f \n", (float)num_zeros/(float)
    (N*N));
    return 0;
}