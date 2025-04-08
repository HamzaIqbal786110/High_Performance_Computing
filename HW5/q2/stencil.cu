#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 32

// ORIGINAL ALGORITHM
// float a[n][n][n], b[n][n][n];
// for (i=1; i<n-1; i++)
// {
//     for (j=1; j<n-1; j++)
//     {
//         for (k=1; k<n-1; k++) 
//         {
//             a[i][j][k]=0.75*(b[i-1][j][k]+b[i+1][j][k]+b[i][j-1][k] + b[i][j+1][k]+b[i][j][k-1]+b[i][j][k+1]);
//         }
//     }
// }

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}


__global__ void stencil(float *a, float* b, int n)
{
    int i = blockIdx.x + blockDim.x + threadIdx.x;
    int j = blockIdx.y + blockDim.y + threadIdx.y;
    int k = blockIdx.z + blockDim.z + threadIdx.z;
    int idx = (i * n * n) + (j * n) + k;
    if (i > 0 && i < n-1 && j > 0 && j < n-1 && k > 0 && k < n-1)
    {
        int id1 = ((i-1) * n * n) + (j * n) + (k);
        int id2 = ((i+1) * n * n) + (j * n) + (k);
        int id3 = (i * n * n) + ((j-1) * n) + (k);
        int id4 = (i * n * n) + ((j+1) * n) + (k);
        int id5 = (i * n * n) + (j * n) + (k-1);
        int id6 = (i * n * n) + (j * n) + (k+1);
        a[idx] = 0.75*(b[id1]+b[id2]+b[id3]+b[id4]+b[id5]+b[id6]);
    }

}


int main(int argc, char *argv[])
{

    double start, end, time_taken;
    
    float* a;
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