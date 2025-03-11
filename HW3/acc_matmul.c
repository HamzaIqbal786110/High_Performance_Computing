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
    __m512d c0, c1, c2, c3, c4, c5, c6, c7;
    
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
    while(B * B < CACHE_SIZE / 24)
    {
        B *= 2;
    }
    printf("B = %i\n", B);
    printf("starting dense matrix multiply \n");
    
    start = CLOCK();
    // Tiling
    //omp_set_num_threads(LOOPS);
    #pragma omp parallel for collapse(3) private(jj,kk,i,j,k,c0,c1,c2,c3,c4,c5,c6,c7)
    for (l=0; l<LOOPS; l++) 
    {
        for(jj=0; jj<N; jj+=B)
        {
            for(kk=0; kk<N; kk+=B)
            {
                // Using AVX512
                // Source https://github.com/romz-pl/matrix-matrix-multiply/blob/main/src/dgemm_avx512.cpp
                for(i=0;i+7<N;i+=8)
                {
                    for(j=jj; j<jj+B && j<N; j+=8)
                    {
                        c0 = _mm512_load_pd(&c[i][j]);
                        c1 = _mm512_load_pd(&c[i+1][j]);
                        c2 = _mm512_load_pd(&c[i+2][j]);
                        c3 = _mm512_load_pd(&c[i+3][j]);
                        c4 = _mm512_load_pd(&c[i+4][j]);
                        c5 = _mm512_load_pd(&c[i+5][j]);
                        c6 = _mm512_load_pd(&c[i+6][j]);
                        c7 = _mm512_load_pd(&c[i+7][j]);
                        for(k=kk; k<kk+B && k<N; k++)
                        {
                            __m512d bb = _mm512_load_pd(&b[k][j]);
                            __m512d aa0 = _mm512_set1_pd(a[i][k]);
                            __m512d aa1 = _mm512_set1_pd(a[i+1][k]);
                            __m512d aa2 = _mm512_set1_pd(a[i+2][k]);
                            __m512d aa3 = _mm512_set1_pd(a[i+3][k]);
                            __m512d aa4 = _mm512_set1_pd(a[i+4][k]);
                            __m512d aa5 = _mm512_set1_pd(a[i+5][k]);
                            __m512d aa6 = _mm512_set1_pd(a[i+6][k]);
                            __m512d aa7 = _mm512_set1_pd(a[i+7][k]);
                            c0 = _mm512_fmadd_pd(aa0, bb, c0);
                            c1 = _mm512_fmadd_pd(aa1, bb, c1);
                            c2 = _mm512_fmadd_pd(aa2, bb, c2);
                            c3 = _mm512_fmadd_pd(aa3, bb, c3);
                            c4 = _mm512_fmadd_pd(aa4, bb, c4);
                            c5 = _mm512_fmadd_pd(aa5, bb, c5);
                            c6 = _mm512_fmadd_pd(aa6, bb, c6);
                            c7 = _mm512_fmadd_pd(aa7, bb, c7);
                        }
                        _mm512_store_pd(&c[i][j], c0);
                        _mm512_store_pd(&c[i+1][j], c1);
                        _mm512_store_pd(&c[i+2][j], c2);
                        _mm512_store_pd(&c[i+3][j], c3);
                        _mm512_store_pd(&c[i+4][j], c4);
                        _mm512_store_pd(&c[i+5][j], c5);
                        _mm512_store_pd(&c[i+6][j], c6);
                        _mm512_store_pd(&c[i+7][j], c7);
                    }
                }
            }
        }
    }
    
    double scale = 1.0 / LOOPS;
    __m512d factor = _mm512_set1_pd(scale);
    #pragma omp for private(j, c0) collapse(2)
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
    
    /* initialize a sparse matrix */
    num_zeros = 0;
    int total_num_zeros = 0;
    int max_nzeros = 0;
    for(i=0; i<N; i++)
    {
        num_zeros = 0;
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
        total_num_zeros += num_zeros;
        max_nzeros = (num_zeros > max_nzeros) ? num_zeros : max_nzeros; 
    }

    printf("\n\nstarting sparse matrix multiply \n");
    
    start = CLOCK();
    
    double **a_val = malloc(N * sizeof(double *));
    int **a_col_ind = malloc(N * sizeof(int *));
    double **b_val = malloc(N * sizeof(double *));
    int **b_col_ind = malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) 
    {
        a_val[i] = malloc(max_nzeros * sizeof(double));
        a_col_ind[i] = malloc(max_nzeros * sizeof(int));
        b_val[i] = malloc(max_nzeros * sizeof(double));
        b_col_ind[i] = malloc(max_nzeros * sizeof(int));
    }

    int ak, bk;
    #pragma omp parallel for private(i, j, ak, bk) num_threads(4)
    for(i = 0; i < N; i++)
    {
        ak = 0;
        bk = 0;
        for(j = 0; j < N; j++)
        {
            c[i][j] = 0;
            if(a[i][j] != 0)
            {
                a_val[i][ak] = a[i][j];
                a_col_ind[i][ak] = j;
                ak++;
            }
            if(b[i][j] != 0)
            {
                b_val[i][bk] = b[i][j];
                b_col_ind[i][bk] = j;
                bk++;
            }
        }
        for(; ak < max_nzeros; ak++)
        {
            a_val[i][ak] = 0;
            a_col_ind[i][ak] = -1;
        }
        for(; bk < max_nzeros; bk++)
        {
            b_val[i][bk] = 0;
            b_col_ind[i][bk] = -1;
        }
    }

    int a_col, b_col;
    double a_value, b_value;
    //#pragma omp parallel for collapse(2) private(i, ak, a_col, bk, a_value, b_value)
    for (l=0; l<LOOPS; l++) 
    {
        for(i = 0; i < N; i++)
        {
            for(ak=0; ak<max_nzeros && a_col_ind[i][ak] != -1; ak++)
            {
                int a_col = a_col_ind[i][ak];
                double a_value = a_val[i][ak];
                for (bk = 0; bk < max_nzeros && b_col_ind[a_col][bk] != -1; bk++) 
                {
                        int b_col = b_col_ind[a_col][bk];
                        double b_value = b_val[a_col][bk];
                        c[i][b_col] += a_value * b_value;
                }
            }
        }
    }
    
    #pragma omp for private(j, c0) collapse(2)
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
    for (int i = 0; i < N; i++) 
    {
        free(a_val[i]);
        free(a_col_ind[i]);
        free(b_val[i]);
        free(b_col_ind[i]);
    }
    free(a_val);
    free(a_col_ind);
    free(b_val);
    free(b_col_ind);

    total = finish-start;
    printf("A result %g \n", c[7][8]); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with sparse matrices = %f ms\n",
    total);
    printf("The sparsity of the a and b matrices = %f \n", (float)total_num_zeros/(float)
    (N*N));
    return 0;
}