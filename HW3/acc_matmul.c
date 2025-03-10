#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
#include <omp.h>

#define N 512
#define LOOPS 10
#define CACHE_SIZE 1887000 / 56 //Approxmately


double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}


int main()
{
    double a[N][N] __attribute__((aligned(64))); /* input matrix */
    double b[N][N] __attribute__((aligned(64))); /* input matrix */
    double c[N][N] __attribute__((aligned(64))); /* result matrix */
    int i,j,k,kk,jj,ii,l, num_zeros;
    double start, finish, total;
    
    // printf("OMP_PROC_BIND: %s\n", getenv("OMP_PROC_BIND"));
    // printf("OMP_PLACES: %s\n", getenv("OMP_PLACES"));

    /* initialize a dense matrix */
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            a[i][j] = (double)(i+j);
            b[i][j] = (double)(i-j);
            c[i][j] = 0.0;
        }
    }
    
    // Block Size Calculation
    int B = 1;
    while(B * B < CACHE_SIZE / 192)
    {
        B *= 2;
    }

    B = 16;
    printf("B = %i\n", B);
    printf("starting dense matrix multiply \n");
    
    start = CLOCK();
    // Tiling
    //omp_set_num_threads(LOOPS);
    
    for (l=0; l<LOOPS; l++) 
    {
        #pragma omp parallel for num_threads(16) collapse(1) private(jj,kk,i,j,k) schedule(static)
        for(jj=0; jj<N; jj+=B)
        {
            for(kk=0; kk<N; kk+=B)
            {
                // Using AVX512
                // Source https://github.com/romz-pl/matrix-matrix-multiply/blob/main/src/dgemm_avx512.cpp
                for(i=0;i<N;i++)
                {
                    for(j=jj; j<jj+B && j<N; j+=8)
                    {
                        __m512d c0 = _mm512_loadu_pd(&c[i][j]); //_mm512_setzero_pd();
                        for(k=kk; k<kk+B && k<N; k++)
                        {
                            __m512d bb = _mm512_loadu_pd(&b[k][j]);
                            __m512d aa = _mm512_set1_pd(a[i][k]);
                            c0 = _mm512_fmadd_pd(aa, bb, c0);
                        }
                        _mm512_storeu_pd(&c[i][j], c0);
                    }
                }
            }
        }
    }
    
    double scale = 1.0 / LOOPS;
    __m512d factor = _mm512_set1_pd(scale);
    #pragma omp parallel for private(j) num_threads(32) collapse(2)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j += 8)
        {
            __m512d c0 = _mm512_loadu_pd(&c[i][j]);
            c0 = _mm512_mul_pd(c0, factor);
            _mm512_storeu_pd(&c[i][j], c0);
        }
    }

    finish = CLOCK();
    total = finish-start;
    printf("a result %g \n", c[7][8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with dense matrices = %f ms\n",
    total);
    return 0;
    /* initialize a sparse matrix */
    num_zeros = 0;
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            if ((i<j)&&(i%2>0))
            {
                a[i][j] = (double)(i+j);
                b[i][j] = (double)(i-j);
            }
            else
            {
                num_zeros++;
                a[i][j] = 0.0;
                b[i][j] = 0.0;
            }
        }
    }

    printf("starting sparse matrix multiply \n");
    
    start = CLOCK();
    for (l=0; l<LOOPS; l++) 
    {
        for(i=0; i<N; i++)
        {
            for(j=0; j<N; j++)
            {
                c[i][j] = 0.0;
                for(k=0; k<N; k++)
                {
                    c[i][j] = c[i][j] + a[i][k] * b[k][j];
                }
            }
        }
    }
    finish = CLOCK();
    
    total = finish-start;
    printf("A result %g \n", c[7][8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with sparse matrices = %f ms\n",
    total);
    printf("The sparsity of the a and b matrices = %f \n", (float)num_zeros/(float)
    (N*N));
    return 0;
}