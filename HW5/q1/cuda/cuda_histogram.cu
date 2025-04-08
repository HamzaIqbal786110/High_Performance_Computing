#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
//#include <device_launch_parameter.h>

// #define NUM_RANDS 8000000
#define RAND_RANGE 100000
#define NUM_BINS 64
#define BLOCK_SIZE 128

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}


__global__ void binner(int *nums, int* results, int nums_per_thread, int NUM_RANDS)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tIdx = threadIdx.x;
    
    __shared__ int block_results[BLOCK_SIZE][NUM_BINS];

    int thread_results[NUM_BINS] = {0};
    int start = tid * nums_per_thread;
    int end = min(NUM_RANDS, start+nums_per_thread);
    for(int i = start; i < end; i++)
    {
        int bin = min((nums[i] / (RAND_RANGE / NUM_BINS)), (NUM_BINS - 1));
        thread_results[bin]++;
    }

    for(int b = 0; b < NUM_BINS; b++)
    {
        block_results[tIdx][b] = thread_results[b];
    }
    __syncthreads();

    if(tIdx < NUM_BINS)
    {
        int sum = 0;
        for(int t = 0; t < blockDim.x; t++)
        {
            sum += block_results[t][tIdx];
        }
        atomicAdd(&results[tIdx], sum);
    }

}


int main(int argc, char *argv[])
{

    double start, end, time_taken;
    
    int* nums;
    int* results = (int*)calloc(NUM_BINS, sizeof(int));
   
    int NUM_RANDS = 8388608;
    if (argc > 1) NUM_RANDS = atoi(argv[1]);

    nums = (int*)malloc(sizeof(int) * NUM_RANDS);
    srand(time(NULL));
    for(int i = 0; i < NUM_RANDS; i++)
    {
        nums[i] = (rand() % RAND_RANGE) + 1;
    }


    start = CLOCK();
    int *nums_d, *results_d;
    cudaMalloc(&nums_d, NUM_RANDS * sizeof(int));
    cudaMalloc(&results_d, NUM_BINS * sizeof(int));
    cudaMemcpy(nums_d, nums, NUM_RANDS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(results_d, 0, NUM_BINS * sizeof(int));

    int grid_size = (NUM_RANDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_threads = grid_size * BLOCK_SIZE;
    int nums_per_thread = (NUM_RANDS + total_threads - 1) / total_threads;
    binner<<<grid_size, BLOCK_SIZE>>>(nums_d, results_d, nums_per_thread, NUM_RANDS);
    cudaDeviceSynchronize();
    cudaMemcpy(results, results_d, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    end = CLOCK();
   
    for(int i = 0; i < NUM_BINS; i++)
    {
        printf("Bin %i = %i\n", i, results[i]);
    }
    
    time_taken = end - start;

    printf("Time taken = %f ms \n", time_taken);
    return(0);
}