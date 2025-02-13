#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>
#include<semaphore.h>
#include<unistd.h>

#define NUM_PHILOSOPHERS 5
double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

sem_t forks[NUM_PHILOSOPHERS];

void* philosopher(void *arg)
{
    int philosopher_num = *(int*)arg;
    int lork = philosopher_num;
    int rork = (philosopher_num + 1) % NUM_PHILOSOPHERS;

    for(int i = 0; i < 5; i++)
    {
        switch(philosopher_num)
        {
            case 0:
                printf("The man who runs in front of the car gets tired, the man who runs behind the car gets exhausted. - Philosopher %d\n\n", philosopher_num);
                sleep(10);
                break;
            case 1:
                printf("You don't need a parachute to go skydiving, but you need one to go skydiving twice. - Philosopher %d\n\n", philosopher_num);
                sleep(10);
                break;
            case 2:
                printf("Trying to define yourself is like trying to bite your own teeth. - Philosopher %d\n\n", philosopher_num);
                sleep(10);
                break;
            case 3:
                printf("Everybody has a plan until they get punched in the mouth. - Philosopher %d\n\n", philosopher_num);
                sleep(10);
                break;
            case 4:
                printf("Cauliflower is nothing but cabbage with a college education - Philosopher %d\n\n", philosopher_num);
                sleep(10);
                break;
            default:
                printf("Ma balls is hot - Jon Jones \n\n");
                sleep(10);
                break;
        }   
        if(philosopher_num % 2 == 0)
        {
            sem_wait(&forks[lork]);
            sem_wait(&forks[rork]);
            printf("Philosopher %d is chowing down. \n\n", philosopher_num);
            sem_post(&forks[rork]);
            sem_post(&forks[lork]);
        }
        else
        {
            sem_wait(&forks[rork]);
            sem_wait(&forks[lork]);
            printf("Philosopher %d is chowing down. \n\n", philosopher_num);
            sem_post(&forks[lork]);
            sem_post(&forks[rork]);
        }
    }
    
}

int main()
{
    pthread_t philosophers[NUM_PHILOSOPHERS];
    int ids[NUM_PHILOSOPHERS];

    for(int i = 0; i < NUM_PHILOSOPHERS; i++)
    {
        sem_init(&forks[i], 0, 1);
        ids[i] = i;
    }
    for(int i = 0; i < NUM_PHILOSOPHERS; i++)
    {
        pthread_create(&philosophers[i], NULL, philosopher, (void*)&ids[i]);
    }
    for(int i = 0; i < NUM_PHILOSOPHERS; i++)
    {
        pthread_join(philosophers[i], NULL);
    }

    return 0;
}