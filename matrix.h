
#pragma once

#include <stdlib.h>

#define BLOCK_SIZE 4096
#define BLOCK_TOTAL_SIZE BLOCK_SIZE*BLOCK_SIZE


// Structured like this so we can copy over contiguous memory to the gpu
typedef struct {
	// Number of elements
	// m rows
	int m;
	// n cols
	int n;
	// m rows of blocks
	int m_blocks;
	// n cols of blocks
	int n_blocks;
	// flattened m_blocks x n_blocks each with BLOCK_SIZExBLOCK_SIZE elements
	double** blocks;
} block_matrix_t;


void bm_create(block_matrix_t *bm, int m, int n);
void bm_fill_matrix(block_matrix_t *bm, int seed);
void bm_free(block_matrix_t *bm);



