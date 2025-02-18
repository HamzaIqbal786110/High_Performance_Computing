HW 2 
All the executable files are included in the directory so you should be able to just run them.

If not then for each of the files here is the associated compilation command:

graph_coloring.c     |  gcc graph_coloring.c -o graph_coloring
leibniz.c            |  gcc leibniz.c -o leibniz
montecarlo.c         |  gcc montecarlo.c -lm -o montecarlo
omp_graph_coloring.c |  gcc -g -fopenmp omp_graph_coloring.c -o omp_graph_coloring
omp_leibniz.c        |  gcc omp_leibniz.c -fopenmp -o omp_leibniz
omp_montecarlo.c     |  gcc omp_montecarlo.c -fopenmp -lm -o omp_montecarlo
philosophers.c       |  gcc philosophers.c -lpthread -o philosophers
pthread_leibniz.c    |  gcc pthread_leibniz.c -lpthread -o pthread_leibniz
pthead_montecarlo.c  |  gcc pthread_montecarlo.c -lpthread -lm -o pthread_montecarlo

