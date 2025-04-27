#include <mpi.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>
#include "gemm.h"
#include "mpi_message.h"
#include "matrix.h"

#define SERVER_RANK 0
#define CLIENT_RANK 1
#define NUM_ROLES 2
// CUDA, CLIENT, and SERVER functions
#define TEST_ROWS BLOCK_SIZE * 4
#define TEST_COLS BLOCK_SIZE * 4
#define TEST_SEED 1

void print_hostname(int world_rank) {
    char hostname[HOST_NAME_MAX + 1];

    if (gethostname(hostname, sizeof(hostname)) == 0) {
        printf("CLIENT %d: Hostname: %s\n", world_rank, hostname);
    } else {
        perror("gethostname");
    }
}


// Global vars
MPI_Comm shmcomm;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
	printf("SERVER: m_blocks %d n_blocks %d\n", final_bm.m_blocks, final_bm.n_blocks);
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

			clock_t before = clock();
			MPI_Recv(final_bm.blocks[dst_block], BLOCK_TOTAL_SIZE, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);

			clock_t difference = clock() - before;
			int msec = difference * 1000 / CLOCKS_PER_SEC;
			printf("SERVER: RECV BLOCK TIME %d ms\n", msec);
		}
	}
	bm_free(&final_bm);
}

// Increment rows and cols of 2 tiles
void client_inc_tile_ctrs(const block_matrix_t* bm1, const block_matrix_t* bm2, int* row_ctr1, int* col_ctr1, int* row_ctr2, int* col_ctr2){
	*row_ctr1 = (*row_ctr1 + 1) % bm1->m_blocks;
	*col_ctr2 = (*col_ctr2 + 1) % bm2->n_blocks;

}

void client(int world_rank){
	print_hostname(world_rank);
	block_matrix_t bm1;
	bm_create(&bm1, TEST_ROWS, TEST_COLS);

	block_matrix_t bm2;
	bm_create(&bm2, TEST_ROWS, TEST_COLS);

	// Random initialization... if this was real life we'd just have a pointer or something
	bm_fill_matrix(&bm1, TEST_SEED); 

	bm_fill_matrix(&bm2, TEST_SEED); 
	// Freeing gpu
	printf("CLIENT %d: Freeing GPU\n", world_rank);
	cudaFree(0);
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
		const int scol = start_row_col[1];
		int row_ctr1 = srow;
		int col_ctr1 = scol;
		int row_ctr2 = srow;
		int col_ctr2 = scol;

		printf("CLIENT %d: Received WL ROW: %d COL: %d\n", world_rank, srow, scol);

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
		printf("CLIENT %d BEGINNING CUDA INITIALIZATION\n", world_rank);
		gpuErrchk(cudaMalloc(&tile_A1, BLOCK_TOTAL_SIZE * sizeof(double)));
		gpuErrchk(cudaMalloc(&tile_B1, BLOCK_TOTAL_SIZE * sizeof(double)));
		gpuErrchk(cudaMalloc(&tile_A2, BLOCK_TOTAL_SIZE * sizeof(double)));
		gpuErrchk(cudaMalloc(&tile_B2, BLOCK_TOTAL_SIZE * sizeof(double)));
		gpuErrchk(cudaMalloc(&tile_C , BLOCK_TOTAL_SIZE * sizeof(double)));
		printf("CLIENT %d FINISHED CUDA INITIALIZATION\n", world_rank);

		// Initialize output to zeroes
		cudaMemcpyAsync(tile_C, output_block, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);

		// Handles copying data, commanding cuda process to copy, and 
		for(int i = 0; i < bm1.m_blocks; i++){
			printf("CLIENT %d Multiplying tiles %d,%d and %d,%d\n", world_rank, row_ctr1, col_ctr1, row_ctr2, col_ctr2);
			// Determines which buffer to prefetch in
			int prefetch_buffer = (i + 1) % 2;
			// copy data over with and cudaMemCpy (non async)
			// row_ctr * bm.n_blocks + col_ctr;
			if(i == 0){
				double* block_addrA = bm1.blocks[row_ctr1 * bm1.n_blocks + col_ctr1];
				double* block_addrB = bm2.blocks[row_ctr2 * bm2.n_blocks + col_ctr2];
				clock_t before = clock();
				printf("CLIENT %d: COPYING BLOCK ADDR %d and %d TO GPU\n", world_rank, row_ctr1 * bm1.n_blocks + col_ctr1, row_ctr2 * bm2.n_blocks + col_ctr2);
				cudaMemcpy(tile_A1, block_addrA, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(tile_B1, block_addrB, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);

				clock_t difference = clock() - before;
				int msec = difference * 1000 / CLOCKS_PER_SEC;
				printf("CLIENT %d: GPU COPY TIME %d ms\n", world_rank, msec);
			}

			client_inc_tile_ctrs(&bm1, &bm2, &row_ctr1, &col_ctr2, &row_ctr2, &col_ctr2);
			// Prefetch blocks with cudaMemCpyAsync and launch kernel on other tiles

			double* block_addrA = bm1.blocks[row_ctr1 * bm1.n_blocks + col_ctr1];
			double* block_addrB = bm2.blocks[row_ctr2 * bm2.n_blocks + col_ctr2];
			printf("CLIENT %d: COPYING BLOCK ADDR %d and %d TO GPU\n", world_rank, row_ctr1 * bm1.n_blocks + col_ctr1, row_ctr2 * bm2.n_blocks + col_ctr2);
			if(prefetch_buffer == 0){
				if(i < bm1.m_blocks - 1){
					cudaMemcpyAsync(tile_A1, block_addrA, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
					cudaMemcpyAsync(tile_B1, block_addrB, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
				}

				clock_t before = clock();
				launch_gemm(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, tile_A2, tile_B2, tile_C);
				clock_t difference = clock() - before;
				int msec = difference * 1000 / CLOCKS_PER_SEC;
				printf("CLIENT %d: GPU GEMM TIME %d ms\n",world_rank, msec);
			} else{
				if(i < bm1.m_blocks - 1){
					cudaMemcpyAsync(tile_A2, block_addrA, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
					cudaMemcpyAsync(tile_B2, block_addrB, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyHostToDevice);
				}

				launch_gemm(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, tile_A1, tile_B1, tile_C);
			}
		}
		cudaMemcpy(output_block, tile_C, BLOCK_TOTAL_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
				
		msg[0] = CLIENT_RETURN_WL;
		printf("CLIENT %d: Sending data ping\n", world_rank);
		MPI_Send(msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		printf("CLIENT %d: Sending data block\n", world_rank);
		MPI_Send(output_block, BLOCK_TOTAL_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		cudaFree(&tile_A1);
		cudaFree(&tile_A2);
		cudaFree(&tile_B1);
		cudaFree(&tile_B2);
		cudaFree(&tile_C);

	}
	bm_free(&bm1);
	bm_free(&bm2);
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
		client(world_rank);			
	}
	printf("=== Process %d exiting ===\n", world_rank);
	MPI_Finalize();
}
