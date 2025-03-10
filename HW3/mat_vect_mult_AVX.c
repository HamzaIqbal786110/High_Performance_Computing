#include <stdio.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <time.h>

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

float* mat_vect_mult(const float **mat, const float *vect, size_t size) 
{
    float* result = (float*)malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) 
    {
        result[i] = 0;
        for(int j = 0; j < size; j++)
        {
            result[i] += vect[j] * mat[i][j];
        }
    }
    return result;
}

float* mat_vect_mult_avx512f(const float **a, const float *b, size_t len) 
{
    float* result = (float*)malloc(sizeof(float) * len);
    float sum_arr[len][16];
    __m512 product_vec, a_vec, b_vec;
    for(size_t i = 0; i < len; i++)
    {
         product_vec = _mm512_setzero_ps();
        for (size_t j = 0; j + 16 <= len; j += 16) 
        {
            a_vec = _mm512_loadu_ps(&a[i][j]);
            b_vec = _mm512_loadu_ps(&b[j]);
            product_vec += _mm512_mul_ps(a_vec, b_vec);
        }
        result[i] = _mm512_reduce_add_ps(product_vec);
    }
    return result;
}


int main() 
{
    #define N 64
    double start, finish, total;
    float* result;
    int i;
    float **a = (float**)malloc(N * sizeof(float*));
    float b[N];
    size_t len = sizeof(b)/sizeof(b[0]);
    // size_t len = sizeof(a) / sizeof(a[0]);
    for (i=0; i< N; i++) 
    {
        a[i] = (float*)malloc(N * sizeof(float));
        for(int j = 0; j < N; j++)
        {
            a[i][j] = i + j + 1;
        }
        b[i] = i + 1;
    }
    // start = CLOCK();
    // result = dot_product_avx512f(a, b, len);
    // finish = CLOCK();
    // total = finish-start;
    // printf("Dot product result = %f \n", result); /* prevent dead code elimination */
    // printf("The total time for matrix multiplication with AVX = %f ms\n", total);
    start = CLOCK();
    result = mat_vect_mult((const float**)a, b, len);
    finish = CLOCK();
    total = finish-start;
    for(int i = 0; i < 10; i++)
    {
        printf("result[%i] = %f \n", i, result[i]); /* prevent dead code elimination */
    }
    printf("The total time for matrix multiplication without AVX = %f ms\n", total);
    

    start = CLOCK();
    result = mat_vect_mult_avx512f((const float**)a, b, len);
    finish = CLOCK();
    total = finish-start;
    for(int i = 0; i < 10; i++)
    {
        printf("result[%i] = %f \n", i, result[i]); /* prevent dead code elimination */
    }
    printf("The total time for matrix multiplication with AVX = %f ms\n", total);
    return 0;
}