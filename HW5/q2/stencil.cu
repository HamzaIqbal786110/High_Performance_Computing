#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>


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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
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

    if (argc < 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    size_t total_elems = n * n * n;
    size_t total_bytes = total_elems * sizeof(float);

    // Host memory allocation
    float *a_h = (float*)malloc(total_bytes);
    float *b_h = (float*)malloc(total_bytes);

    // Initialize b_h with random floats
    for (size_t i = 0; i < total_elems; i++) {
        b_h[i] = (float)(rand()) / RAND_MAX;
    }

    // Device memory allocation
    double start = CLOCK();

    float *a_d, *b_d;
    cudaMalloc(&a_d, total_bytes);
    cudaMalloc(&b_d, total_bytes);

    // Copy b to device
    cudaMemcpy(b_d, b_h, total_bytes, cudaMemcpyHostToDevice);
    cudaMemset(a_d, 0, total_bytes);

    // Kernel launch configuration
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y,
                 (n + blockDim.z - 1) / blockDim.z);

    

    // Launch the kernel
    stencil<<<gridDim, blockDim>>>(a_d, b_d, n);
    cudaDeviceSynchronize();

    ;

    // Copy result back
    cudaMemcpy(a_h, a_d, total_bytes, cudaMemcpyDeviceToHost);

    // Print a small slice for verification (optional)
    printf("a[n/2][n/2][n/2] = %f\n", a_h[(n/2)*n*n + (n/2)*n + (n/2)]);
    double end = CLOCK();
    printf("GPU time for N = %d: %f ms\n", n, end - start);
    // Free memory
    cudaFree(a_d);
    cudaFree(b_d);
    free(a_h);
    free(b_h);

    return 0;
}
