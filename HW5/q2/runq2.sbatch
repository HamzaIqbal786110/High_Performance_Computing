#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=stencil_q2
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu
#SBATCH --output=stencil_master.out


OUTFILE="stencil_results.txt"
echo "3D Stencil Benchmark (Naive vs Tiled)" > $OUTFILE
echo "=====================================" >> $OUTFILE
echo "" >> $OUTFILE

# Values of N to test
for N in 16 32 64 128 256; do
    echo "===== Running Naive Stencil: N = $N =====" | tee -a $OUTFILE
    ./stencil $N >> $OUTFILE 2>&1
    echo "----------------------------------------" >> $OUTFILE
    echo "" >> $OUTFILE

    echo "===== Running Tiled Stencil: N = $N =====" | tee -a $OUTFILE
    ./stencil_tiled $N $BLOCK_SIZE >> $OUTFILE 2>&1
    echo "----------------------------------------" >> $OUTFILE
    echo "" >> $OUTFILE
done
