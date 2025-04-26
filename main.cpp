#include <mpi.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "mpi_message.h"
#include "matrix.h"

#define SERVER_RANK 0
#define CLIENT_RANK 1
#define NUM_ROLES 2
// CUDA, CLIENT, and SERVER functions
#define TEST_ROWS BLOCK_SIZE * 8
#define TEST_COLS BLOCK_SIZE * 8
#define TEST_SEED 1

MPI_Comm shmcomm;

// extern kernel
void launch_multiply(const float *a, float *b);

void server_inc_ctrs(const block_matrix_t* bm, int* row_ctr, int* col_ctr){
	if(*row_ctr < 0){
		return;
	}
	*col_ctr = *col_ctr + 1;
	if(*col_ctr >= bm->n_blocks){
		*col_ctr = 0;
		*row_ctr = *row_ctr + 1;	
	}
	if(*row_ctr >= bm->m_blocks){
		*col_ctr = -1;
		*row_ctr = -1;

	}	
}

void server(){
	std::map<int, int> running_wls;
	block_matrix_t final_bm;
	bm_create(&final_bm, TEST_ROWS, TEST_COLS);
	int row_ctr = 0;
	int col_ctr = 0;

	while(row_ctr >= 0 || running_wls.size() > 0){
		int message;
		MPI_Status status;
		MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int source = status.MPI_SOURCE;
		int tag = status.MPI_TAG;
		
		if(message == CLIENT_REQUEST_WL){
			int coords[2] = {row_ctr, col_ctr};
			MPI_Send(coords, 2, MPI_INT, source, tag, MPI_COMM_WORLD);
			running_wls[source] = row_ctr * final_bm.n_blocks + col_ctr;
			printf("SERVER: Client: %d RECIEVED ROW %d COL %d\n", source, row_ctr, col_ctr);
			server_inc_ctrs(&final_bm, &row_ctr, &col_ctr);
		}
		else if(message == CLIENT_RETURN_WL){
			int dst_block = running_wls[source];
			printf("SERVER: Client: %d RECIEVED BLOCK AT %d\n", source, dst_block);
			MPI_Recv(final_bm.blocks[dst_block], BLOCK_TOTAL_SIZE, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
		}
	}
	bm_free(&final_bm);
}

// Increment rows and cols of 2 tiles
void client_inc_tile_ctrs(const block_matrix_t* bm, int* row_ctr1, int* col_ctr1, int* row_ctr2, int* col_ctr2){
	*row_ctr1 = (*row_ctr1 + 1) % bm->m_blocks;
	*col_ctr2 = (*col_ctr2 + 1) % bm->n_blocks;

}

void client(int world_rank, int local_rank){
	block_matrix_t bm;
	bm_create(&bm, BLOCK_SIZE * 3, BLOCK_SIZE * 3);
	// Random initialization... if this was real life we'd just have a pointer or something
	bm_fill_matrix(&bm, TEST_SEED); 
	while(1){
		// Get starting coords
		int msg[1] = {CLIENT_REQUEST_WL};
		printf("CLIENT %d: Requesting WL\n", world_rank);
		MPI_Send(msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

		// Receive Coords
		int start_row_col[2];
		MPI_Status status;
		MPI_Recv(&start_row_col, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// Starting row and col
		const int srow = start_row_col[0];
		const int scol = start_row_col[0];
		int row_ctr1 = srow;
		int col_ctr1 = scol;
		int row_ctr2 = srow;
		int col_ctr2 = scol;

		printf("CLIENT %d: Received WL ROW: %d COL: %d", world_rank, srow, scol);

		// Exit case
		if (srow == -1 || scol == -1) {
			break;	
		}
		
		double output_block[BLOCK_TOTAL_SIZE] = {0};	
		// Allocate 5 buffers, 4 for inputs, 1 for output
		double *tile_A1 = nullptr;
		double *tile_B1 = nullptr;
		double *tile_A2 = nullptr;
		double *tile_B2 = nullptr;
		double *tile_C  = nullptr;
		cudaMalloc(&tile_A1, BLOCK_TOTAL_SIZE * sizeof(double));
		cudaMalloc(&tile_B1, BLOCK_TOTAL_SIZE * sizeof(double));
		cudaMalloc(&tile_A2, BLOCK_TOTAL_SIZE * sizeof(double));
		cudaMalloc(&tile_B2, BLOCK_TOTAL_SIZE * sizeof(double));
		cudaMalloc(&tile_C , BLOCK_TOTAL_SIZE * sizeof(double));

		// Initialize output to zeroes
		cudaMemcpyAsync(tile_C, output_block, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);

		// Handles copying data, commanding cuda process to copy, and 
		for(int i = 0; i < bm.m_blocks; i++){
			// Determines which buffer to prefetch in
			int prefetch_buffer = (i + 1) % 2;
			// copy data over with and cudaMemCpy (non async)
			// row_ctr * bm.n_blocks + col_ctr;
			if(i == 0){
				double* block_addrA = bm.blocks[row_ctr1 * bm.n_blocks + col_ctr1];
				double* block_addrB = bm.blocks[row_ctr2 * bm.n_blocks + col_ctr2];
				cudaMemcpy(tile_A1, block_addrA, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(tile_B1, block_addrB, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
			}

			client_inc_tile_ctrs(&bm, &row_ctr1, &col_ctr2, &row_ctr2, &col_ctr2);
			// Prefetch blocks with cudaMemCpyAsync
			if(!(i >= bm.m_blocks)){

				double* block_addrA = bm.blocks[row_ctr1 * bm.n_blocks + col_ctr1];
				double* block_addrB = bm.blocks[row_ctr2 * bm.n_blocks + col_ctr2];
				if(prefetch_buffer == 0){
					cudaMemcpyAsync(tile_A1, block_addrA, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
					cudaMemcpyAsync(tile_B1, block_addrB, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
				} else{
					cudaMemcpyAsync(tile_A2, block_addrA, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
					cudaMemcpyAsync(tile_B2, block_addrB, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
				}
			}
			// Launch Kernel
		}
	}
	bm_free(&bm);
}

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get local rank
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
	                    MPI_INFO_NULL, &shmcomm);
	int local_rank;
	MPI_Comm_rank(shmcomm, &local_rank);

	// Only 1 master between nodes
	if(world_rank == 0){
		server();			
	}
	else if(local_rank == 1){
		client(world_rank, local_rank);			
	}
	MPI_Finalize();
}
