#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

double leibniz(int n_iter)
{
    double pi = 0;
    #pragma omp parallel for reduction(+:pi)
    for(int i = 0; i < n_iter; i++)
    {
        int denom = (i * 2) + 1;
        if(i % 2 == 0) pi += ((double)4 / denom);
        else pi -= ((double)4 / denom);
    }
    return(pi);
}

int main()
{
    double start, end, runtime;
    int n_iter, num_threads;
    printf("Enter number of times to do the loop:  ");
    scanf("%d", &n_iter);
    printf("\nEnter number of threads to use:  ");
    scanf("%d", &num_threads);

    start = CLOCK();
    srand(time(NULL));
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);
    double pi = leibniz(n_iter);
    end = CLOCK();

    runtime = (end - start) / 1000;

    printf("Computed Value of PI: %4.10f \n", pi);
    printf("Time taken to compute PI: %4.2f s \n", runtime);

    return 0;
}
