#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:05:00
#SBATCH --job-name=run_histogram
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu
#SBATCH --output=run_histogram.out


# Run your CUDA histogram executable
./cuda_histogram