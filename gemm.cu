#include <cuda_runtime.h>
#include <cuda.h>
#include "matrix.h"
#include "gemm.h"

// For gemm0 and gemm1
//#define BLOCK_SIZE_X 32
//#define BLOCK_SIZE_Y 32
//#define BLOCK_SIZE_K 32

// For gemm2+
#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 128
#define BLOCK_SIZE_K 16

#define THREAD_TILE_SIZE_X 8
#define THREAD_TILE_SIZE_Y 8



// Load into tiles with bound checks
__device__ void load_to_shared(const double* A, const double* B, 
		double A_tile[BLOCK_SIZE_Y][BLOCK_SIZE_K], 
		double B_tile[BLOCK_SIZE_K][BLOCK_SIZE_X],
		int tile_idx, int thread_idx, int m, int n, int k){

	constexpr int num_threads = BLOCK_SIZE_X * BLOCK_SIZE_Y;

	// Load A
#pragma unroll
	for(int i = 0; i < (BLOCK_SIZE_Y * BLOCK_SIZE_K + num_threads - 1) / num_threads; i++){
		int A_row_tile_idx = (thread_idx + i * num_threads) / BLOCK_SIZE_K;
		int A_col_tile_idx = (thread_idx + i * num_threads) % BLOCK_SIZE_K;
		int A_row_idx = blockIdx.y * BLOCK_SIZE_Y + A_row_tile_idx;
		int A_col_idx = tile_idx * BLOCK_SIZE_K + A_col_tile_idx;
		double val = 0;

		if(A_row_idx < m && A_col_idx < k){
			val = A[A_row_idx * k + A_col_idx];
		}

		A_tile[A_row_tile_idx][A_col_tile_idx] = val;

	}


	// Load B
#pragma unroll
	for(int i = 0; i < (BLOCK_SIZE_K * BLOCK_SIZE_X + num_threads - 1) / num_threads; i++){
		int B_row_tile_idx = (thread_idx + i * num_threads) / BLOCK_SIZE_X;
		int B_col_tile_idx = (thread_idx + i * num_threads) % BLOCK_SIZE_X;
		int B_row_idx = tile_idx * BLOCK_SIZE_K + B_row_tile_idx;
		int B_col_idx = blockIdx.x * BLOCK_SIZE_X + B_col_tile_idx;
		double val = 0;

		if(B_row_idx < k && B_col_idx < n){
			val = B[B_row_idx * n + B_col_idx];
		}

		B_tile[B_row_tile_idx][B_col_tile_idx] = val;
	}

}

// Basic
__global__ void gemm0(int m, int n, int k, double *A, double *B, double *C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
	int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension
	
	if (row < m && col < n) {
		double sum = 0.0;
		for (int i = 0; i < k; ++i) {
			sum += A[row * k + i] * B[i * n + col];
		}
		C[row * n + col] = sum + C[row * n + col];
	}
}



// Block Tiling
__global__ void gemm1(int m, int n, int k, double *A, double *B, double *C) {

	int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
	int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension

	int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ double A_tile[BLOCK_SIZE_Y][BLOCK_SIZE_K];
	__shared__ double B_tile[BLOCK_SIZE_K][BLOCK_SIZE_X];

	

	int num_tiles = (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;

	double sum = 0;
	for(int tile_idx = 0; tile_idx < num_tiles; tile_idx++){
		// Fill tiles
		load_to_shared(A, B, A_tile, B_tile, tile_idx, thread_idx, m, n, k);		
		__syncthreads();

		// Perform Computation
#pragma unroll
		for(int k_idx = 0; k_idx < BLOCK_SIZE_K; k_idx++){
			sum += A_tile[threadIdx.y][k_idx] * B_tile[k_idx][threadIdx.x];
		}
		__syncthreads();
			
	}
	if(row < m && col < n){
		C[row * n + col] += sum;
	}

	
}

// Thread Tiling
__global__ void gemm2(int m, int n, int k, double *A, double *B, double *C) {

	int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ double A_tile[BLOCK_SIZE_Y][BLOCK_SIZE_K];
	__shared__ double B_tile[BLOCK_SIZE_K][BLOCK_SIZE_X];

	int num_tiles = (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;


	double thread_results [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {0};

	// Intended to be in regs
	double A_vals[THREAD_TILE_SIZE_Y] = {0};

	// Intended to be in regs
	double B_vals[THREAD_TILE_SIZE_X] = {0};


	for(int tile_idx = 0; tile_idx < num_tiles; tile_idx++){
		// Fill tiles
		load_to_shared(A, B, A_tile, B_tile, tile_idx, thread_idx, m, n, k);		
		__syncthreads();

		// Perform Computation
#pragma unroll
		for(int k_idx = 0; k_idx < BLOCK_SIZE_K; k_idx++){
			int A_thread_block_row_idx = (thread_idx / (BLOCK_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y);
			int A_thread_block_col_idx = k_idx;
#pragma unroll
			for(int ttr_idx = 0; ttr_idx < THREAD_TILE_SIZE_Y; ttr_idx++){
				// transpose to prevent bank conflicts
				A_vals[ttr_idx] = A_tile[A_thread_block_row_idx + ttr_idx][A_thread_block_col_idx];
			}

			int B_thread_block_row_idx = k_idx;
			int B_thread_block_col_idx = thread_idx % (BLOCK_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X;

#pragma unroll
			for(int ttc_idx = 0; ttc_idx < THREAD_TILE_SIZE_X; ttc_idx++){
				// transpose to prevent bank conflicts
				B_vals[ttc_idx] = B_tile[B_thread_block_row_idx][B_thread_block_col_idx + ttc_idx];
			}

#pragma unroll
			for(int ttr_idx = 0; ttr_idx < THREAD_TILE_SIZE_Y; ttr_idx++){
				for(int ttc_idx = 0; ttc_idx < THREAD_TILE_SIZE_X; ttc_idx++){
					thread_results[ttr_idx][ttc_idx] += A_vals[ttr_idx] * B_vals[ttc_idx];
				}	
			}

		}
		__syncthreads();
	}

	// Write results back to C
	for(int ttr_idx = 0; ttr_idx < THREAD_TILE_SIZE_Y; ttr_idx++){
		for(int ttc_idx = 0; ttc_idx < THREAD_TILE_SIZE_X; ttc_idx++){
			int C_ri = blockIdx.y * BLOCK_SIZE_Y + threadIdx.x / (BLOCK_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y + ttr_idx;
			int C_ci = blockIdx.x * BLOCK_SIZE_X + threadIdx.x / (BLOCK_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X + ttc_idx;
			if(C_ri < m && C_ci < n){
				C[C_ri * n + C_ci] += thread_results[ttr_idx][ttc_idx];
			}
		}	
	}
}

void launch_gemm(int m, int n, int k, double* A, double* B, double* C) {
	// Launch kernel
	
	//dim3 block_dim(BLOCK_SIZE_X,BLOCK_SIZE_Y,1);
	//dim3 grid_dim(
	//		(n + block_dim.x - 1) / block_dim.x,
	//		(m + block_dim.y - 1) / block_dim.y,
	//		1);
	int threads_per_block = BLOCK_SIZE_X * BLOCK_SIZE_Y / (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y);

	dim3 block_dim(threads_per_block, 1, 1);
	dim3 grid_dim(
			(n + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
			(m + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
			1);
	gemm2<<<grid_dim, block_dim>>>(m, n, k, A, B, C);
	
	// Error check and sync
	cudaPeekAtLastError();
	cudaDeviceSynchronize();
}
