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

float dot_product(const float *vec1, const float *vec2, size_t size) 
{
    double result = 0.0;
    for (int i = 0; i < size; i++) 
    {
        result += vec1[i] * vec2[i];
    }
    return result;
}

float dot_product_avx512f(const float *a, const float *b, size_t len) 
{
    size_t i = 0;
    float sum_arr[16];
    __m512 sum_vec, a_vec, b_vec;
    sum_vec = _mm512_setzero_ps();
    for (; i + 16 <= len; i += 16) 
    {
        a_vec = _mm512_loadu_ps(&a[i]);
        b_vec = _mm512_loadu_ps(&b[i]);
        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    _mm512_storeu_ps(sum_arr, sum_vec);
    float total_sum = 0.0f;
    for (int j = 0; j < 16; j++) 
    {
        total_sum += sum_arr[j];
    }
    for (; i < len; i++) 
    {
        total_sum += a[i] * b[i];
    }
    return total_sum;
}
int main() 
{
    #define N 4096
    double start, finish, total, result;
    int i;
    float a[N], b[N];
    size_t len = sizeof(a)/sizeof(a[0]);
    // size_t len = sizeof(a) / sizeof(a[0]);
    for (i=0; i< N; i++) 
    {
        a[i] = 1.0;
        b[i] = 1.0;
    }
    start = CLOCK();
    result = dot_product_avx512f(a, b, len);
    finish = CLOCK();
    total = finish-start;
    printf("Dot product result = %f \n", result); /* prevent dead code elimination */
    printf("The total time for matrix multiplication with AVX = %f ms\n", total);
    start = CLOCK();
    result = dot_product(a, b, len);
    finish = CLOCK();
    total = finish-start;
    printf("Dot product result = %f \n", result); /* prevent dead code elimination */
    printf("The total time for matrix multiplication without AVX = %f ms\n", total);
    return 0;
}