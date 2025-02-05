#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<pthread.h>

#define MAX_NUM 100000

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

typedef struct
{
    double x;
    double y;
} point;


point generate_point(unsigned int seed)
{
    double rand_x = ((double)rand_r(&seed) / RAND_MAX) * MAX_NUM;
    double rand_y = ((double)rand_r(&seed) / RAND_MAX) * MAX_NUM;
    point rand_point;
    rand_point.x = rand_x;
    rand_point.y = rand_y;

    return(rand_point);
}

double calc_dist(point point1, point point2)
{
    double dist = sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2));
    return(dist);
}

void* monte_carlo(void *arg)
{
    int iter_num = (int)(u_int64_t)arg;
    double radius = (double)MAX_NUM / 2;
    point center;
    center.x = radius;
    center.y = radius;
    int count_in_circle = 0;

    for(int i = 0; i < iter_num; i++)
    {
        unsigned int seed = (time(NULL) * 123456) + (unsigned int)pthread_self() + i;
        point test_point = generate_point(seed);
        double distance = calc_dist(test_point, center);
        if(distance <= radius)
        {
            count_in_circle++;
        }
    }
    double *pi_estimate = malloc(sizeof(double));
    *pi_estimate = ((double)count_in_circle / (double)iter_num) * 4;
    return((void*)pi_estimate);
}

int main()
{
    double start, end, runtime;
    int num_iterations, num_threads;
    printf("Enter number of ""dart throws"":  ");
    scanf("%d", &num_iterations);
    printf("\nEnter number of threads:  ");
    scanf("%d", &num_threads);
    

    start = CLOCK();
    
    srand(time(NULL));
    num_iterations = num_iterations / num_threads;
    pthread_t threads[num_threads];
    for(int i = 0; i < num_threads; i++)
    {
        pthread_create(&threads[i], NULL, monte_carlo, (void *)(u_int64_t)num_iterations);
    }
    double pi = 0;
    double *temp = malloc(sizeof(double));
    for(int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], (void**)&temp);
        pi += *temp;
    }
    free(temp);
    pi = pi / (double)num_threads;

    end = CLOCK();

    runtime = (end - start) / 1000;

    printf("Computed Value of PI: %4.8f \n", pi);
    printf("Time taken to compute PI: %4.2f s \n", runtime);

    return 0;
}