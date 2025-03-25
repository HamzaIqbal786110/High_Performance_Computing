#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define NUM_RANDS 8000000
#define RAND_RANGE 100000
#define NUM_BINS 100

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

int main(int argc, char *argv[])
{
    double start_time, end_time, time_taken;
    start_time = CLOCK();
    int rank, size, name_len;
    char node[MPI_MAX_PROCESSOR_NAME];
    int* nums = NULL;
    int* results = calloc(NUM_BINS, sizeof(int));

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(node, &name_len);

    // Setup for uneven distribution
    int base_chunk = NUM_RANDS / size;
    int extra = NUM_RANDS % size;

    int* sendcounts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_chunk + (i < extra ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_count = sendcounts[rank];
    int* local_nums = malloc(local_count * sizeof(int));

    if(rank == 0)
    {
        nums = (int*)malloc(sizeof(int) * NUM_RANDS);
        srand(time(NULL));
        for(int i = 0; i < NUM_RANDS; i++)
        {
            nums[i] = (rand() % RAND_RANGE) + 1;
        }
    }

    // Replace Bcast with Scatterv
    MPI_Scatterv(nums, sendcounts, displs, MPI_INT, local_nums, local_count, MPI_INT, 0, MPI_COMM_WORLD);

    int bin_size = RAND_RANGE / NUM_BINS;
    int* local_results = calloc(NUM_BINS, sizeof(int));

    for(int i = 0; i < local_count; i++)
    {
        int bin_ind = local_nums[i] / bin_size;
        if (bin_ind >= NUM_BINS) bin_ind = NUM_BINS - 1;
        local_results[bin_ind]++;
    }

    MPI_Reduce(local_results, results, NUM_BINS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if(rank == 0)
    {
        for(int i = 0; i < NUM_BINS; i++)
        {
            printf("Bin %i = %i\n", i, results[i]);
        }
    }

    MPI_Finalize();
    end_time = CLOCK();
    time_taken = end_time - start_time;
    printf("Time taken = %fms \n", time_taken);

    // Cleanup
    free(local_results);
    free(local_nums);
    free(sendcounts);
    free(displs);
    if (rank == 0) {
        free(nums);
        free(results);
    }

    return 0;
}
