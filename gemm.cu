#include <cuda_runtime.h>
#include <cuda.h>
#include "matrix.h"
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

void launch_gemm(int m, int n, int k, double* A, double* B, double* C) {
	// Launch kernel
	
	dim3 blockDim(16,16);
	dim3 gridDim((n + 15) / 16, (m + 15) / 16);
	gemm0<<<gridDim, blockDim>>>(m, n, k, A, B, C);
	
	// Error check and sync
	cudaGetLastError();
	cudaDeviceSynchronize();
}
