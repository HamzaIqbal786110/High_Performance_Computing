#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define BLOCK_SIZE 256

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

__global__ void leibniz(double *pi, int n_iter, int iter_per_thread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * iter_per_thread;
    int end = min(n_iter, start + iter_per_thread);
    double local_pi = 0;
    for(int i = start; i < end; i++)
    {
        local_pi += 4.0 * ((i % 2 == 0) ? 1.0 : -1.0) / (2 * i + 1);
    }
    atomicAdd(pi, local_pi);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <n_iter>\n", argv[0]);
        return 1;
    }

    int n_iter = atoi(argv[1]);

    double *pi_d, pi = 0.0;

    double start = CLOCK();
    
    cudaMalloc(&pi_d, sizeof(double));
    cudaMemset(pi_d, 0, sizeof(double));

    int grid_size = (n_iter + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_threads = grid_size * BLOCK_SIZE;
    int iter_per_thread = (n_iter + total_threads - 1) / total_threads;

    
    leibniz<<<grid_size, BLOCK_SIZE>>>(pi_d, n_iter, iter_per_thread);
    cudaDeviceSynchronize();

    cudaMemcpy(&pi, pi_d, sizeof(double), cudaMemcpyDeviceToHost);

    double end = CLOCK();

    printf("Computed Value of PI with %d iterations: %.10f\n", n_iter, pi);
    printf("GPU Time Taken: %.4f ms\n", end - start);

    cudaFree(pi_d);
    return 0;
}