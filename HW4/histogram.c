#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define NUM_RANDS 8000000
#define RAND_RANGE 100000
#define NUM_BINS 128

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

int main(int argc, char *argv[])
{

    double start, end, time_taken;
    start = CLOCK();
    int* nums;
    int* results = calloc(NUM_BINS, sizeof(int));
   
   

    nums = (int*)malloc(sizeof(int) * NUM_RANDS);
    srand(time(NULL));
    for(int i = 0; i < NUM_RANDS; i++)
    {
        nums[i] = (rand() % RAND_RANGE) + 1;
    }

    for(int i = 0; i < NUM_RANDS; i++)
    {
        int bin_ind = nums[i] / (RAND_RANGE/NUM_BINS);
        if (bin_ind >= NUM_BINS) bin_ind = NUM_BINS - 1;
        results[bin_ind]++;
    }
    
   
    for(int i = 0; i < NUM_BINS; i++)
    {
        printf("Bin %i = %i\n", i, results[i]);
    }
    end = CLOCK();
    time_taken = end - start;

    printf("Time taken = %f ms \n", time_taken);
    return(0);
}