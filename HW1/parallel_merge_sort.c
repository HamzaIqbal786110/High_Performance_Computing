/* SOURCE: https://gist.github.com/mycodeschool/9678029 */
/* Merge sort in C */
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>

#define ARR_LEN 10000
#define NUM_THREADS 8

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}



struct args
{
	int* A;
	int n;
};


void Merge(int *A,int *L,int leftCount,int *R,int rightCount) 
{
	int i,j,k;
	i = 0; j = 0; k = 0;

	while(i<leftCount && j< rightCount) 
	{
		if(L[i]  < R[j]) A[k++] = L[i++];
		else A[k++] = R[j++];
	}
	while(i < leftCount) A[k++] = L[i++];
	while(j < rightCount) A[k++] = R[j++];
}

void Final_Merge(int **A, struct args **thread_arrs, int n)
{
	
	if(NUM_THREADS == 1)
	{
		*A = thread_arrs[0]->A;
	}
	else
	{
		while(n > 1)
		{
			int new_n = (n + 1) / 2;
			int** new_arrs = (int**)malloc(new_n * sizeof(int*));
			int* new_sizes = (int*)malloc(new_n * sizeof(int));
			int k = 0;
			for(int i = 0; i < n; i+= 2)
			{
				
				if(i + 1 < n)
				{
					new_arrs[k] = (int*)malloc((thread_arrs[i]->n + thread_arrs[i+1]->n) * sizeof(int));
					Merge(new_arrs[k], thread_arrs[i]->A, thread_arrs[i]->n, thread_arrs[i+1]->A, thread_arrs[i+1]->n);
					new_sizes[k] = thread_arrs[i]->n + thread_arrs[i+1]->n;
				}
				else
				{
					new_arrs[k] = thread_arrs[i]->A;
					new_sizes[k] = thread_arrs[i]->n;
				}
				k++;
			}
			
			thread_arrs = (struct args**)malloc(sizeof(struct args*) * new_n);
			for(int i = 0; i < new_n; i++)
			{
				thread_arrs[i] = malloc(sizeof(struct args));
				thread_arrs[i]->A = new_arrs[i];
				thread_arrs[i]->n = new_sizes[i];
			}
			n = new_n;
			free(new_arrs);
			free(new_sizes);
		}
		*A = thread_arrs[0]->A;

		free(thread_arrs[0]);
		free(thread_arrs);

	}
}


// Recursive function to sort an array of integers. 
void* MergeSort(void *input) 
{
	int *A = ((struct args*)input) -> A;
	int n = ((struct args*)input) -> n;
	
	int mid,i, *L, *R;
	if(n < 2) return NULL; // base condition. If the array has less than two element, do nothing. 

	mid = (n/2);  // find the mid index. 

	L = (int*)malloc(mid*sizeof(int)); 
	R = (int*)malloc((n-mid)*sizeof(int)); 
	
	for(i = 0; i<mid; i++) L[i] = A[i]; // creating left subarray
	for(i = mid; i<n; i++) R[i-mid] = A[i]; // creating right subarray

	struct args *Left = (struct args *)malloc(sizeof(struct args));
	Left->A = L;
	Left->n = mid;

	struct args *Right = (struct args *)malloc(sizeof(struct args));
	Right->A = R;
	Right->n = n-mid;

	MergeSort((void *)Left);  // sorting the left subarray
	MergeSort((void *)Right);  // sorting the right subarray
	
	Merge(A,L,mid,R,n-mid);  // Merging L and R into A as sorted list.
    
	free(L);
    free(R);
	free(Left);
	free(Right);

	return NULL;
}

int main() 
{
	/* Code to test the MergeSort function. */
    double start, finish, total;
	
	int jump = ARR_LEN / NUM_THREADS;
	int extra = ARR_LEN % NUM_THREADS;

	int *A = (int*)malloc(sizeof(int) * ARR_LEN);
	int **thread_arrs = (int**)malloc(sizeof(int*) * NUM_THREADS);
	pthread_t threads[NUM_THREADS];
	struct args *arg_arr[NUM_THREADS];
    for(int i = 0; i < ARR_LEN; i++)
    {
        A[i] = rand() / 10000;
    }

	// Calling merge sort to sort the array. 
    start = CLOCK();
	for(int i = 0; i<NUM_THREADS; i++)
	{
		arg_arr[i] = (struct args*)malloc(sizeof(struct args));
		
		if(i != NUM_THREADS - 1)
		{
			arg_arr[i]->n = jump;
			thread_arrs[i] = (int*)malloc(sizeof(int) * jump);
		}
		else
		{
			arg_arr[i]->n = jump+extra;
			thread_arrs[i] = (int*)malloc(sizeof(int) * (jump + extra));
		}
		for(int j = 0; j < (arg_arr[i]->n); j++) thread_arrs[i][j] = A[(i*jump) + j];
		arg_arr[i]->A = thread_arrs[i];
		pthread_create(&threads[i], NULL, MergeSort, (void *)arg_arr[i]);
	}
	for(int i = 0; i<NUM_THREADS; i++) pthread_join(threads[i], NULL);
    Final_Merge(&A, arg_arr, NUM_THREADS);
	finish = CLOCK();
    total = finish - start;
	
	
	for(int i = 0; i < NUM_THREADS; i++) free(arg_arr[i]);
	
	//printing all elements in the array once its sorted.
	//for(int i = 0;i < ARR_LEN;i++) printf("%d \n",A[i]);
    printf("\nTime for Parallel MergeSort = %4.2f ms \n", total);
	for(int i = 0; i < NUM_THREADS; i++)
	{
		free(thread_arrs[i]);
	}
	free(thread_arrs);
	free(A);
	return 0;
}