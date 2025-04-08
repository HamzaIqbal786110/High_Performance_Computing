#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=leibniz_pi
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu
#SBATCH --output=leibniz_master.out

OUTFILE="leibniz_results.txt"
echo "CUDA π Benchmark (Double vs Single Precision)" > $OUTFILE
echo "=============================================" >> $OUTFILE
echo "" >> $OUTFILE

# Values of n_iter to test (powers of 2)
for EXP in {10..23}; do
    N=$((2 ** EXP))

    echo "===== Running DOUBLE PRECISION: n_iter = $N =====" | tee -a $OUTFILE
    ./cuda_leibniz_dp $N >> $OUTFILE 2>&1
    echo "-------------------------------------------------" >> $OUTFILE
    echo "" >> $OUTFILE

    echo "===== Running SINGLE PRECISION: n_iter = $N =====" | tee -a $OUTFILE
    ./cuda_leibniz_sp $N >> $OUTFILE 2>&1
    echo "-------------------------------------------------" >> $OUTFILE
    echo "" >> $OUTFILE
done
