#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>


void bm_create(block_matrix_t *bm, int m, int n){
	bm->m = m;
	bm->n = n;
	bm->m_blocks = m % BLOCK_SIZE == 0 ? m / BLOCK_SIZE : m / BLOCK_SIZE + 1;
	bm->n_blocks = n % BLOCK_SIZE == 0 ? n / BLOCK_SIZE : n / BLOCK_SIZE + 1;
	double** blocks = (double**) malloc(sizeof(double*) * bm->m_blocks * bm->n_blocks);
	for(int i = 0; i < bm->m_blocks * bm->n_blocks; i++){
		blocks[i] = (double*) malloc(BLOCK_TOTAL_SIZE * sizeof(double));	
	}
	bm->blocks = blocks;
}
void bm_fill_matrix(block_matrix_t *bm, int seed){
	srand(seed);
	for(int i = 0; i < bm->m_blocks * bm->n_blocks; i++){
		for(int j = 0; j < BLOCK_SIZE * BLOCK_SIZE; j++){
			bm->blocks[i][j] = (double) (rand() % 200 - 100);
		}
	}
}
void bm_free(block_matrix_t *bm){
	for(int i = 0; i < bm->n_blocks * bm->m_blocks; i++){
		free(bm->blocks[i]);	
	}
	free(bm->blocks);
}
