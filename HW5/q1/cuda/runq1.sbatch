#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=hw5q1
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu
#SBATCH --output=hw5q1_master.out

# Output file
OUTFILE="hw5q1_all_output.txt"
echo "CUDA Histogram Benchmarking" > $OUTFILE
echo "============================" >> $OUTFILE

# Loop through powers of 2 from 2^12 to 2^23
for ((exp=12; exp<=23; exp++)); do
    N=$((2 ** exp))
    echo "Running with N = $N" | tee -a $OUTFILE
    ./cuda_histogram $N >> $OUTFILE 2>&1
    echo "----------------------------------------" >> $OUTFILE
done
