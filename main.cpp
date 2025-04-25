#include <mpi.h>
#include <map>
#include <stdio.h>
#include "mpi_message.h"
#include "matrix.h"

// extern kernel
void launch_multiply(const float *a, float *b);


#define SERVER_RANK 0
#define CLIENT_RANK 1
#define CUDA_RANK 2
#define NUM_ROLES 3
// CUDA, CLIENT, and SERVER functions

void inc_ctrs(const block_matrix_t* bm, int* row_ctr, int* col_ctr){
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
	bm_create(&final_bm, BLOCK_SIZE * 3, BLOCK_SIZE * 3);
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
			inc_ctrs(&final_bm, &row_ctr, &col_ctr);
		}
		else if(message == CLIENT_RETURN_WL){
			int dst_block = running_wls[source];
			MPI_Recv(final_bm.blocks[dst_block], BLOCK_TOTAL_SIZE, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
		}
	}
	bm_free(&final_bm);
}



void client(){}


void cuda(){}


int main(int argc, char** argv){
	MPI_Init(&argc, &argv);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get local rank
	MPI_Comm shmcomm;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
	                    MPI_INFO_NULL, &shmcomm);
	int local_rank;
	MPI_Comm_rank(shmcomm, &local_rank);

	// Only 1 master between nodes
	if(world_rank == 0){
		server();			
	}
	else if(local_rank == 1){
		client();			
	}
	else if(local_rank == 2){
		cuda();			
	}
	MPI_Finalize();

}
