#include <cuda_runtime.h>
#include <stdio.h>

#define M 256
#define N 256
#define K 256

__global__ void gemm_mnk_add(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

int main() {
    printf("HIHIHI!\n\n");
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // Fill with example data
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; ++i) h_C[i] = 2.0f; // so we see accumulation

    float *d_A, *d_B, *d_C;
    printf("before cudaMalloc\n");
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    printf("before Memcpy\n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    float alpha = 1.0f, beta = 1.0f;

    printf("before kernel\n");
    gemm_mnk_add<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

    printf("before memcpy back\n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("after memcpy back\n");
    // Check result
    printf("C[0][0] = %f (should be %f)\n", h_C[0], alpha * K + beta * 2.0f);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}

