Instructions to run each program 

float_taylor.c
    gcc float_taylor.c -lm -o float_taylor
    ./float_taylor
    - It will ask for inputs

double_taylor.c
    gcc double_taylor.c -lm -o double_taylor
    ./double_taylor
    - It will ask for inputs

dot-product-AVX.c
    Make sure you are on a cascadelake node otherwise AVX wont work
    gcc dot-product-AVX.c -mavx512f -o dot-product-AVX
    ./dot-product-AVX

mat_vect_mult_AVX.c
    Make sure you are on a cascadelake node otherwise AVX wont work
    gcc mat_vect_mult_AVX.c -mavx512f -o mat_vect_mult_AVX
    ./mat_vect_mult_AVX 

matmul.c
    gcc matmul.c -o matmul
    ./matmul.c

accelerated_matmul.c
    Make sure you are on a cascadelake node otherwise AVX wont work
    gcc accelerated_matmul.c -fopenmp -mavx512f -O3 -march=native -o accelerated_matmul
    ./accelerated_matmul
    - Note: Based on the cache heirarchy it will work best if the CACHE_SIZE definition at the top is modified to match.

blas_simple.c
    module load OpenBLAS/0.3.29
    gcc blas_simple.c -lopenblas -o blas_simple
    ./blas_simple

blas_matmul.c
    module load OpenBLAS/0.3.29
    gcc blas_matmul.c -lopenblas -o blas_matmul
    ./blas_matmul