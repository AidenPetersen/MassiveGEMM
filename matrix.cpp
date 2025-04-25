#include "matrix.h"
#include <stdlib.h>


void bm_create(block_matrix_t *bm, int m, int n){
	bm->m = m;
	bm->n = n;
	bm->m_blocks = m % BLOCK_SIZE == 0 ? m / BLOCK_SIZE : m / BLOCK_SIZE + 1;
	bm->n_blocks = n % BLOCK_SIZE == 0 ? n / BLOCK_SIZE : n / BLOCK_SIZE + 1;
	blocks = (double**) malloc(sizeof(double*) * m_blocks * n_blocks);
	for(int i = 0; i < m_blocks * n_blocks; i++){
		blocks[i] = malloc(BLOCK_SIZE * sizeof(double));	
	}
	bm->blocks = blocks;

}
void bm_fill_matrix(block_matrix_t *block_matrix){
	for(int i = 0; i < bm->m_blocks * bm->n_blocks; i++){
		for(int j = 0; j < BLOCK_SIZE * BLOCK_SIZE; j++){
			bm->blocks[i][j] = (double) (rand() % 200 - 100);
		}
	}
}
void bm_free(block_matrix_t *bm){
	for(int i = 0; i < n_blocks * m_blocks; i++){
		free(bm->blocks[i]);	
	}
	free(bm->blocks);
}
