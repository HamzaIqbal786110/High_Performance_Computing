for ((exp=12; exp<=23; exp++)); do
    N=$((2 ** exp))
    echo "Running for N = $N"
    ./omp_histogram $N > omp_histogram_$N.out
done
