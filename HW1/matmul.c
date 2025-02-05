#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 // This is to benchmark a memory bound program

#define MAT_SIZE 1500
void matrix_multiply(u_int64_t **A, u_int64_t **B, u_int64_t **C, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            C[i][j] = 0;
            for(int k = 0; k < n; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main()
{
    u_int64_t **A, **B, **C;
    A = (u_int64_t **)malloc(MAT_SIZE * sizeof(u_int64_t *));
    B = (u_int64_t **)malloc(MAT_SIZE * sizeof(u_int64_t *));
    C = (u_int64_t **)malloc(MAT_SIZE * sizeof(u_int64_t *));
    for (int i = 0; i < MAT_SIZE; i++) {
        A[i] = (u_int64_t *)malloc(MAT_SIZE * sizeof(u_int64_t));
        B[i] = (u_int64_t *)malloc(MAT_SIZE * sizeof(u_int64_t));
        C[i] = (u_int64_t *)malloc(MAT_SIZE * sizeof(u_int64_t));
    }
    srand(time(NULL));  // Seed for randomness
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            A[i][j] = rand() % 100;  // Random values between 0 and 99
            B[i][j] = rand() % 100;
        }
    }

    matrix_multiply(A, B, C, MAT_SIZE);

    for (int i = 0; i < MAT_SIZE; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;

}