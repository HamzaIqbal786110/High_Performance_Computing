#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<stdint.h>
#include<time.h>
#include<math.h>

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

float fac(int n)
{
    float result = 1;
    for(int i = n; i > 0; i--)
    {
        result *= i;
    }
    return(result);
}

float taylor_series(float x, int n_terms)
{
    float result = 0;
    for(int i = 0; i < n_terms; i++)
    {
        result = (i % 2 == 0) ? (result + pow(x, ((2*i) + 1)) / fac(((2*i )+ 1))) : (result - pow(x, ((2*i) + 1)) / fac(((2*i )+ 1)));
    }
    return(result);
}

int main()
{
    float x;
    int n_terms;

    printf("Enter x: ");
    scanf("%f", &x);
    printf("Enter num terms: ");
    scanf("%i", &n_terms);

    double start = CLOCK();
    float sin_x = taylor_series(x, n_terms);
    double end = CLOCK();

    double time = end - start;

    printf("Sin(%3.10f) = %3.10f\n", x, sin_x);
    printf("Time taken: %3.10f\n", time);

    return(0);
}