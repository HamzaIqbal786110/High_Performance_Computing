#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<stdint.h>
#include<time.h>
#include<math.h>


#define BITSET_SIZE(N) (((N) + 7) / 8)

typedef struct
{
    uint8_t num_nodes;
    // This is an array that will contain all the nodes that are adjacent 
    // near each other between indices specified in offsets
    uint8_t *edges;
    // The offsets to find the adjacent nodes to node n
    // EX: node 3, to find its adjacent nodes you would find the following:
    // indices = offsets[i], offsets[i+1] - 1
    // adjacent nodes are found on edges between the indices
    uint8_t *offsets;
    uint8_t *colors;
} grph;

typedef struct
{
    uint8_t node;
    uint8_t degree;
} nodeDegree;

typedef struct
{
    size_t size;
    uint8_t *bits;
} bitset;

// Initialize a bitset
bitset *bitset_create(size_t num_bits) 
{
    bitset *b = malloc(sizeof(bitset));
    b->size = num_bits;
    b->bits = calloc(BITSET_SIZE(num_bits), sizeof(uint8_t));
    return b;
}

// Set a bit (add element)
void bitset_add(bitset *b, uint8_t value) 
{
    b->bits[value / 8] |= (1 << (value % 8));
}

// Check if a bit is set (contains element)
bool bitset_contains(bitset *b, uint8_t value) 
{
    return (b->bits[value / 8] & (1 << (value % 8))) != 0;
}

// Clear a bit (remove element)
void bitset_remove(bitset *b, uint8_t value) 
{
    b->bits[value / 8] &= ~(1 << (value % 8));
}

// Free bitset
void bitset_destroy(bitset *b) 
{
    free(b->bits);
    free(b);
}

uint8_t find_degree(uint8_t node, grph* graph)
{
    uint8_t degree = graph->offsets[node+1] - graph->offsets[node];
    return(degree);
}


int compare_nodes(const void *a, const void *b)
{
    return(((nodeDegree*)b)->degree - ((nodeDegree*)a)->degree);
}

void sort_nodes_by_degree(grph *graph, uint8_t *sorted_nodes)
{
    nodeDegree node_degrees[graph->num_nodes];
    for(uint8_t i = 0; i < graph->num_nodes; i++)
    {
        node_degrees[i].node = i;
        node_degrees[i].degree = find_degree(i, graph);
    }
    qsort(node_degrees, graph->num_nodes, sizeof(nodeDegree), compare_nodes);
    for(uint8_t i = 0; i < graph->num_nodes; i++)
    {
        sorted_nodes[i] = node_degrees[i].node;
    }
}

void add_neighbors_bitset(bitset *b, uint8_t node, grph* graph)
{
    for(uint8_t i = graph->offsets[node]; i < (graph->offsets[node + 1] - 1); i++)
    {
        bitset_add(b, graph->edges[i]);
    }
}

bool check_edge(uint8_t node1, uint8_t node2, grph* graph)
{
    for(int i = graph->offsets[node1]; i < (graph->offsets[node1+1] - 1); i++)
    {
        if(graph->edges[i] == node2) return(true);
    }
    return(false);
}

bool check_colored(grph* graph)
{
    for(uint8_t i = 0; i < graph->num_nodes; i++)
    {
        if(!graph->colors[i]) return(false);
    }
    return(true);
}

// Using Welsh Powell Algorithm
uint8_t color_graph(grph* graph)
{
    bool all_colored = 0;
    uint8_t *sorted_nodes = malloc(sizeof(uint8_t) * graph->num_nodes);
    sort_nodes_by_degree(graph, sorted_nodes);
    uint8_t color = 0;
    bitset *visited = bitset_create(graph->num_nodes);
    while(!all_colored)
    {
        color++;
        bitset *visited_inner = bitset_create(graph->num_nodes);
        for(uint8_t i = 0; i < graph->num_nodes; i++)
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
    grph graph = 
    {
        .num_nodes = 30,
        .edges = (uint8_t[]) 
        {
            1, 5, 6,           // 0
            0, 2, 6, 7,        // 1
            1, 7, 8,           // 2
            4, 9, 10,          // 3
            3, 9, 10,          // 4
            0, 6, 11,          // 5
            0, 1, 5, 7, 12,    // 6
            1, 2, 6, 8, 13,    // 7
            2, 7, 9, 14,       // 8
            3, 4, 8, 10, 14,   // 9
            3, 4, 9, 15, 16,   // 10
            5, 12, 17,         // 11
            6, 11, 13, 17, 18, // 12
            7, 12, 14, 18, 19, // 13
            8, 9, 13, 19, 20,  // 14
            10, 16, 20,        // 15
            10, 15, 20, 25,    // 16
            11, 12, 18, 21,    // 17
            12, 13, 17, 19, 22,// 18
            13, 14, 18, 20, 23,// 19
            14, 15, 16, 19, 24,// 20
            17, 26,            // 21
            18, 27, 23,        // 22
            19, 22, 24, 28,    // 23
            20, 23, 25, 29,    // 24
            16, 24, 29,        // 25
            21, 27,            // 26
            22, 26, 28,        // 27
            23, 27, 29,        // 28
            24, 25, 28         // 29
        },
        .offsets = (uint8_t[]) 
        {
            0,  3,  7, 10, 13, 16, 19, 24, 29, 33,
            38, 41, 46, 51, 56, 59, 63, 66, 71, 76,
            81, 83, 86, 90, 94, 97, 99, 102, 105, 108, 111
        },
        .colors = calloc(30, sizeof(uint8_t))
    };

    uint8_t chromatic_number = color_graph(&graph);
    printf("The Chromatic Number of the Graph is: %d\n", chromatic_number);
    return(0);
}