#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=hw5q1
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu
#SBATCH --output=hw5q1_master.out


# Loop through powers of 2 from 2^12 to 2^23
for ((exp=12; exp<=23; exp++)); do
    N=$((2 ** exp))
    echo "Running with N = $N"
    ./cuda_histogram $N > hw5q1_${N}.out 2>&1
done