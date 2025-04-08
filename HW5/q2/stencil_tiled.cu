#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 10
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


__global__ void stencil_tiled(float *a, float* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;
    int idx = (i * n * n) + (j * n) + k;

    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    if (i < n && j < n && k < n) 
    {
        // Center of tile
        tile[tx][ty][tz] = b[(i * n * n) + (j * n) + k];

        // Boundaries of tile
        if (threadIdx.x == 0 && i > 0) tile[tx - 1][ty][tz] = b[((i - 1) * n * n) + (j * n) + k];
        if (threadIdx.x == BLOCK_SIZE - 1 && i < n - 1) tile[tx + 1][ty][tz] = b[((i + 1) * n * n) + (j * n) + k];

        if (threadIdx.y == 0 && j > 0) tile[tx][ty - 1][tz] = b[(i * n * n) + ((j - 1) * n) + k];
        if (threadIdx.y == BLOCK_SIZE - 1 && j < n - 1) tile[tx][ty + 1][tz] = b[(i * n * n) + ((j + 1) * n) + k];

        if (threadIdx.z == 0 && k > 0) tile[tx][ty][tz - 1] = b[(i * n * n) + (j * n) + (k - 1)];
        if (threadIdx.z == BLOCK_SIZE - 1 && k < n - 1) tile[tx][ty][tz + 1] = b[(i * n * n) + (j * n) + (k + 1)];

    }

    if (i > 0 && i < n-1 && j > 0 && j < n-1 && k > 0 && k < n-1)
    {
        a[idx] = 0.75*( tile[tx - 1][ty][tz] + 
                        tile[tx + 1][ty][tz] + 
                        tile[tx][ty - 1][tz] + 
                        tile[tx][ty + 1][tz] + 
                        tile[tx][ty][tz - 1] + 
                        tile[tx][ty][tz + 1]);
    }

}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <N> <BLOCK_SIZE>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    printf("Running stencil with N = %d, BLOCK_SIZE = %d\n", n, BLOCK_SIZE);

    size_t total_elems = n * n * n;
    size_t total_bytes = total_elems * sizeof(float);

    // Host memory allocation
    float *a_h = (float*)malloc(total_bytes);
    float *b_h = (float*)malloc(total_bytes);

    // Initialize b_h with random values
    for (size_t i = 0; i < total_elems; i++) {
        b_h[i] = (float)(rand()) / RAND_MAX;
    }

    double start = CLOCK();

    float *a_d, *b_d;
    cudaMalloc(&a_d, total_bytes);
    cudaMalloc(&b_d, total_bytes);

    cudaMemcpy(b_d, b_h, total_bytes, cudaMemcpyHostToDevice);
    cudaMemset(a_d, 0, total_bytes);

    // Configure launch
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    

    stencil_tiled<<<gridDim, blockDim>>>(a_d, b_d, n);
    cudaDeviceSynchronize();
    cudaMemcpy(a_h, a_d, total_bytes, cudaMemcpyDeviceToHost);

    double end = CLOCK();

    printf("Sample value: a[n/2][n/2][n/2] = %f\n", a_h[(n/2)*n*n + (n/2)*n + (n/2)]);
    printf("GPU time for N = %d and BLOCK_SIZE = %d: %f ms\n", n, BLOCK_SIZE, end - start);

    cudaFree(a_d);
    cudaFree(b_d);
    free(a_h);
    free(b_h);

    return 0;
}