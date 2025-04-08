#!/bin/bash

OUTPUT_FILE="omp_histogram_results.txt"
echo "OMP Histogram Benchmarking" > $OUTPUT_FILE
echo "===========================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for ((exp=12; exp<=23; exp++)); do
    N=$((2 ** exp))
    echo "Running for N = $N" | tee -a $OUTPUT_FILE
    ./omp_histogram $N >> $OUTPUT_FILE 2>&1
    echo "--------------------------------------" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
done
