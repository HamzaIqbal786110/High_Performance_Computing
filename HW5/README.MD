How to Run Everything

Question 1

cd q1
cd cuda
module load cuda
nvcc cuda_histogram.cu -o cuda_histogram
sbatch runq1.sbatch
cd ../omp
gcc omp_histogram.c -fopenmp -o omp_histogram
./run_omp_histogram.sh



Question 2
cd q2
module load cuda
nvcc stencil.cu -o stencil
nvcc stencil_tiled.cu -o stencil_tiled
sbatch runq2.sbatch
