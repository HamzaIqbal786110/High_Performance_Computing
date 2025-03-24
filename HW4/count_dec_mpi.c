#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size, name_len, count, tag = 0, sender;
    char node[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(node, &name_len);

    // First process initializes the counter
    if (rank == 0) 
    {
        count = 1;
        printf("Count = %d from process %d at node %s\n", count, rank, node);
        fflush(stdout);
        if (size > 1) 
        {
            MPI_Send(&count, 1, MPI_INT, rank + 1, tag, MPI_COMM_WORLD);
        }
    } 
    while(1) 
    {
        MPI_Recv(&count, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        tag = status.MPI_TAG;
        sender = status.MPI_SOURCE;

        if(tag == 0)
        {
            count++;
            printf("Count = %d from process %d at node %s\n", count, rank, node);
            fflush(stdout);
            if(count == size && rank == size - 1)
            {
                fflush(stdout);
                tag = 1;
                count -= 2;
                if (rank > 0)
                {
                    fflush(stdout);
                    MPI_Send(&count, 1, MPI_INT, rank - 1, tag, MPI_COMM_WORLD);
                    fflush(stdout);
                }
            }
            else
            {
                MPI_Send(&count, 1, MPI_INT, rank + 1, tag, MPI_COMM_WORLD);
            }
        }
        else
        {
            printf("Count = %d from process %d at node %s\n", count, rank, node);
            fflush(stdout);
            count -= 2;
            if(count >= 0 && rank > 0)
            {
                MPI_Send(&count, 1, MPI_INT, rank - 1, tag, MPI_COMM_WORLD);
            }
            
            if (rank == 0 || count < 0) {
                break;
            }
        }
        
    }

    MPI_Finalize();
    return 0;
}
