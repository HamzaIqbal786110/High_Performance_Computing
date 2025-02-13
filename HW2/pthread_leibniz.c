#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

struct arg
{
    int start;
    int end;
};

void* leibniz(void* arg)
{
    int start = ((struct arg*)arg) -> start;
    int end = ((struct arg*)arg) -> end;

    double *pi = malloc(sizeof(double));
    *pi = 0;
    for(int i = start; i < end; i++)
    {
        int denom = (i * 2) + 1;
        if(i % 2 == 0) *pi += ((double)4 / denom);
        else *pi -= ((double)4 / denom);
    }
    return(pi);
}

int main()
{
    double start, end, runtime;
    int n_iter, num_threads;
    printf("Enter number of times to do the loop:  ");
    scanf("%d", &n_iter);
    printf("\nEnter number of threads to use:  ");
    scanf("%d", &num_threads);
    
    start = CLOCK();
    srand(time(NULL));
    pthread_t thread_arr[num_threads];
    struct arg *arg_arr[num_threads];
    int chunk_size = n_iter / num_threads;
    int last_chunk_size = (n_iter / num_threads) + (n_iter % num_threads);
    for(int i = 0; i < num_threads; i++)
    {
        arg_arr[i] = (struct arg*)malloc(sizeof(struct arg));
        arg_arr[i] -> start = (i * chunk_size) + i;
        arg_arr[i] -> end = (i == num_threads - 1) ? (arg_arr[i]->start + last_chunk_size) : (arg_arr[i]->start + chunk_size);
        pthread_create(&thread_arr[i], NULL, leibniz, (void*)arg_arr[i]);
    }
    double pi = 0;
    for(int i = 0; i < num_threads; i++)
    {
        double *temp;
        pthread_join(thread_arr[i], (void**)&temp);
        pi += *temp;
        free(temp);
        free(arg_arr[i]);
    }
    end = CLOCK();

    runtime = (end - start) / 1000;

    printf("Computed Value of PI: %4.10f \n", pi);
    printf("Time taken to compute PI: %4.2f s \n", runtime);

    return 0;
}
