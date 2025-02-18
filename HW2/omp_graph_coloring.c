#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<stdint.h>
#include<time.h>
#include<math.h>
#include<omp.h>
#include<string.h>
#include<x86intrin.h>


#define BITSET_SIZE(N) (((N) + 7) / 8)
#define ENCODE_EDGE(u, v) ((uint64_t)(u) << 32 | (uint64_t)(v))
#define DECODE_U(edge) ((uint32_t)((edge) >> 32))
#define DECODE_V(edge) ((uint32_t)((edge) & 0xFFFFFFFF))



typedef struct
{
    uint32_t num_nodes;
    // This is an array that will contain all the nodes that are adjacent 
    // near each other between indices specified in offsets
    uint32_t *edges;
    // The offsets to find the adjacent nodes to node n
    // EX: node 3, to find its adjacent nodes you would find the following:
    // indices = offsets[i], offsets[i+1] - 1
    // adjacent nodes are found on edges between the indices
    uint32_t *offsets;
    uint16_t *colors;
} grph;

typedef struct
{
    uint32_t node;
    uint16_t degree;
} nodeDegree;

typedef struct
{
    size_t size;
    uint8_t *bits;
} bitset;

typedef struct 
{
    uint64_t s[4]; // PRNG state
} xoshiro256_state;

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

void seed_xoshiro256(xoshiro256_state *state, uint64_t seed) 
{
    for (int i = 0; i < 4; i++)
        state->s[i] = seed = (seed ^ 0x9e3779b97f4a7c15) * 0xbf58476d1ce4e5b9;
}

uint32_t xoshiro256_next(xoshiro256_state *state) 
{
    uint64_t *s = state->s;
    uint64_t result = (s[1] * 5) << 7;

    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> (64 - 45));

    return ((uint32_t)(result >> 32));
}

// Initialize a bitset
bitset *bitset_create(size_t num_bits) 
{
    bitset *b = malloc(sizeof(bitset));
    b->size = num_bits;
    b->bits = calloc(BITSET_SIZE(num_bits), sizeof(uint8_t));
    return b;
}

// Set a bit (add element)
void bitset_add(bitset *b, uint32_t value) 
{
    #pragma omp atomic
    b->bits[value / 8] |= (1 << (value % 8));
}

// Check if a bit is set (contains element)
bool bitset_contains(bitset *b, uint32_t value) 
{
    return (b->bits[value / 8] & (1 << (value % 8))) != 0;
}

// Clear a bit (remove element)
void bitset_remove(bitset *b, uint32_t value) 
{
    b->bits[value / 8] &= ~(1 << (value % 8));
}

// Free bitset
void bitset_destroy(bitset *b) 
{
    free(b->bits);
    free(b);
}

grph* generate_graph(uint32_t num_nodes, uint64_t num_edges) {
    if (num_edges < num_nodes - 1) 
    {
        uint64_t balls = (num_nodes - 1);
        printf("%u\n", num_edges);
        printf("%u\n", balls);
        printf("Error: Not enough edges to form a connected graph!\n");
        return NULL;
    }

    if(num_edges > ((uint64_t)(num_nodes * ((uint64_t)num_nodes - 1)) / 2))
    {
        
        printf("Error: Too many edges for the given number of nodes! \n");
        return NULL;
    }

    grph *graph = malloc(sizeof(grph));
    graph->num_nodes = num_nodes;
    graph->edges = malloc(sizeof(uint32_t) * (2 * num_edges));
    graph->offsets = calloc(num_nodes + 1, sizeof(uint64_t));
    graph->colors = calloc(num_nodes, sizeof(uint16_t));

    uint64_t edge_count = 0;
    uint64_t extra_edges = num_edges - (num_nodes - 1);

    // Step 1: Generate Spanning Tree (Ensures connectivity)
    uint32_t *shuffled = malloc(num_nodes * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_nodes; i++) shuffled[i] = i;

    for (uint32_t i = num_nodes - 1; i > 0; i--) 
    {
        uint32_t j = rand() % (i + 1);
        uint32_t temp = shuffled[i];
        shuffled[i] = shuffled[j];
        shuffled[j] = temp;
    }

    for (uint32_t i = 1; i < num_nodes; i++) 
    {
        uint32_t u = shuffled[i - 1];
        uint32_t v = shuffled[i];
        graph->edges[edge_count++] = v;
        graph->edges[edge_count++] = u;
        graph->offsets[u + 1]++;
        graph->offsets[v + 1]++;
    }

    free(shuffled);

    // Step 2: Add Random Extra Edges Using Reservoir Sampling
    size_t total_possible_edges = ((uint64_t)num_nodes * (num_nodes - 1)) / 2;
    bitset *edge_tracker = bitset_create(total_possible_edges);

    edge_count = 0;
    while (edge_count < extra_edges) {
        uint32_t u = rand() % num_nodes;
        uint32_t v = rand() % num_nodes;
        
        if (u == v) continue; // No self-loops
        if (u < v) { uint32_t temp = u; u = v; v = temp; } // Ensure u > v for unique encoding

        uint64_t edge_index = (uint64_t)u * (u - 1) / 2 + v;
        
        if (bitset_contains(edge_tracker, edge_index)) continue; // Already sampled

        bitset_add(edge_tracker, edge_index); // Mark as used
        graph->edges[edge_count++] = u;
        graph->edges[edge_count++] = v;
        graph->offsets[u + 1]++;
        graph->offsets[v + 1]++;
    }

    bitset_destroy(edge_tracker); // Clean up

    // Step 4: Compute Prefix Sum for Offsets
    for (uint32_t i = 1; i <= num_nodes; i++) {
        graph->offsets[i] += graph->offsets[i - 1];
    }

    return graph;
}


uint16_t find_degree(uint64_t node, grph* graph)
{
    uint16_t degree = graph->offsets[node+1] - graph->offsets[node];
    return(degree);
}


int compare_nodes(const void *a, const void *b)
{
    return(((nodeDegree*)b)->degree - ((nodeDegree*)a)->degree);
}

void sort_nodes_by_degree(grph *graph, uint32_t *sorted_nodes)
{
    nodeDegree node_degrees[graph->num_nodes];
    #pragma omp parallel for
    for(uint32_t i = 0; i < graph->num_nodes; i++)
    {
        node_degrees[i].node = i;
        node_degrees[i].degree = find_degree(i, graph);
    }
    qsort(node_degrees, graph->num_nodes, sizeof(nodeDegree), compare_nodes);
    #pragma omp paralel for
    for(uint32_t i = 0; i < graph->num_nodes; i++)
    {
        sorted_nodes[i] = node_degrees[i].node;
    }
}

void add_neighbors_bitset(bitset *b, uint32_t node, grph* graph)
{
    for(uint64_t i = graph->offsets[node]; i < (graph->offsets[node + 1] - 1); i++)
    {
        bitset_add(b, graph->edges[i]);
    }
}

bool check_colored(grph* graph)
{
    for(uint32_t i = 0; i < graph->num_nodes; i++)
    {
        if(!graph->colors[i]) return(false);
    }
    return(true);
}

// Using Welsh Powell Algorithm
uint16_t color_graph(grph* graph)
{
    bool all_colored = 0;
    uint32_t *sorted_nodes = malloc(sizeof(uint32_t) * graph->num_nodes);
    sort_nodes_by_degree(graph, sorted_nodes);
    uint16_t color = 0;
    bitset *visited = bitset_create(graph->num_nodes);
    while(!all_colored)
    {
        color++;
        bitset *visited_inner = bitset_create(graph->num_nodes);
        #pragma omp parallel for
        for(uint32_t i = 0; i < graph->num_nodes; i++)
        {
            if(!graph->colors[sorted_nodes[i]] && !bitset_contains(visited, sorted_nodes[i]) && !bitset_contains(visited_inner, sorted_nodes[i]))
            {
                graph->colors[sorted_nodes[i]] = color;
                bitset_add(visited, sorted_nodes[i]);
                bitset_add(visited_inner, sorted_nodes[i]);
                add_neighbors_bitset(visited_inner, sorted_nodes[i], graph);
            }
        }
        bitset_destroy(visited_inner);
        all_colored = check_colored(graph);
    }
    bitset_destroy(visited);
    return(color);
}

int main()
{
    
    uint32_t nodes;
    uint64_t edges;
    uint16_t num_threads;
    double start1, start2, end1, end2, total1, total2;
    printf("Enter the number of nodes in the graph: ");
    scanf("%u", &nodes);
    printf("Enter the number of edges in the graph: ");
    scanf("%lu", &edges);
    printf("Enter the number of threads to use: ");
    scanf("%hu", &num_threads);

    start1 = CLOCK();
    omp_set_num_threads(num_threads);
    grph *graph = generate_graph(nodes, edges);
    end1 = CLOCK();
    total1 = end1 - start1;
    printf("Time taken to generate graph: %3.5f ms\n", total1);
    start2 = CLOCK();
    uint64_t chromatic_number = color_graph(graph);
    end2 = CLOCK();
    total2 = end2 - start2;
    printf("The Chromatic Number of the Graph is: %d\n", chromatic_number);
    printf("Time taken to compute chromatic number: %3.5f ms\n", total2);
    return(0);
}