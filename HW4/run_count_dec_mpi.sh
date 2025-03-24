#!/bin/bash
#SBATCH --verbose
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=Hamza_Iqbal_Job
#SBATCH --mem=100G
#SBATCH --partition=courses
#SBATCH --output=count_dec_mpi.out
$SRUN mpirun -mca btl_bas:wq:quUie_warn_component_unused 0 /home/iqbal.ha/High_Performance_Computing/HW4/count_dec_mpi
