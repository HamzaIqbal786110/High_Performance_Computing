#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define RAND_RANGE 100000
#define NUM_BINS 64
#define min(a,b) ((a) < (b) ? (a) : (b))

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

int main(int argc, char *argv[])
{
    int NUM_RANDS = 8000000;
    if (argc > 1) {
        NUM_RANDS = atoi(argv[1]);
    }

    double start, end, time_taken;
    start = CLOCK();

    int* nums = (int*)malloc(sizeof(int) * NUM_RANDS);
    int* results = (int*)calloc(NUM_BINS, sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < NUM_RANDS; i++) {
        nums[i] = (rand() % RAND_RANGE) + 1;
    }

    // Array reduction requires OpenMP 4.5+
    #pragma omp parallel for reduction(+:results[:NUM_BINS])
    for (int i = 0; i < NUM_RANDS; i++) 
    {
        int bin = min((nums[i] / (RAND_RANGE / NUM_BINS)), (NUM_BINS - 1));
        results[bin]++;
    }

    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %i = %i\n", i, results[i]);
    }

    end = CLOCK();
    time_taken = end - start;

    printf("Time taken = %f ms \n", time_taken);
    return 0;
}
