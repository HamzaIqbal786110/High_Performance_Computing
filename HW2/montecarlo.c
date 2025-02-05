#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

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


point generate_point()
{
    double rand_x = ((double)rand() / RAND_MAX) * MAX_NUM;
    double rand_y = ((double)rand() / RAND_MAX) * MAX_NUM;
    point rand_point;
    rand_point.x = rand_x;
    rand_point.y = rand_y;

    return(rand_point);
}

double calc_dist(point point1, point point2)
{
    double dist = sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2));
}

double monte_carlo(int iter_num)
{
    double radius = (double)MAX_NUM / 2;
    point center;
    center.x = radius;
    center.y = radius;
    int count_in_circle = 0;

    for(int i = 0; i < iter_num; i++)
    {
        point test_point = generate_point();
        double distance = calc_dist(test_point, center);
        if(distance <= radius)
        {
            count_in_circle++;
        }
    }
    double pi_estimate = ((double)count_in_circle / (double)iter_num) * 4;
    return(pi_estimate);
}

int main()
{
    double start, end, runtime;
    int num_iterations;
    printf("Enter number of ""dart throws"":  ");
    scanf("%d", &num_iterations);

    start = CLOCK();
    srand(time(NULL));
    double pi = monte_carlo(num_iterations);
    end = CLOCK();

    runtime = (end - start) / 1000;

    printf("Computed Value of PI: %4.5f \n", pi);
    printf("Time taken to compute PI: %4.2f s \n", runtime);

    return 0;
}