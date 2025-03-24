#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size, name_len, count;
    char node[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(node, &name_len);

    // First process initializes the counter
    if (rank == 0) {
        count = 1;
        printf("Count = %d from process %d at node %s\n", count, rank, node);
        fflush(stdout);
        if (size > 1) {
            MPI_Send(&count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    } else {
        // Receive from the previous rank
        MPI_Recv(&count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        count++;
        printf("Count = %d from process %d at node %s\n", count, rank, node);
        fflush(stdout);
        if (rank < size - 1) {
            MPI_Send(&count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
