Instructions for running each of the programs

Before running anything make sure that OpenMPI is loaded, if not run:
module load OpenMPI/4.1.6

All programs were run using sbatch with slurm.

count_mpi.c
mpicc count_mpi.c -o count_mpi
sbatch run_count_mpi.sh
Outputs will be printed to: count_mpi.out


count_dec_mpi.c
mpicc count_dec_mpi.c -o count_dec_mpi
sbatch run_count_dec_mpi.sh
Outputs will be printed to: count_dec_mpi.out


histogram.c
gcc histogram.c -o histogram
sbatch run_histogram.sh
Outputs will be printed to: histogram.out


histogram_mpi.c
mpicc histogram_mpi.c -o histogram_mpi
sbatch run_histogram_mpi.sh
Outputs will be printed to: histogram_mpi.out

To change bin size change the Macro in line 8
#define NUM_BINS 100

To change number of nodes change line 3 in run_histogram_mpi.sh
#SBATCH --nodes=2
Change nodes to the desired number of nodes

To change the number of processes per node change line 4 in run_histogram_mpi.sh
#SBATCH --ntasks-per-node=16
Change ntasks-per-node to the desired number of tasks per node