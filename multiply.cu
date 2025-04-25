#include <cuda_runtime.h>
#include <cuda.h>

__global__ void __multiply__ (const float *a, float *b, int n) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        b[i] *= a[i];
    }
}

extern "C" void launch_multiply(const float *a, float *b, int n) {
    float *a_gpu = nullptr;
    float *b_gpu = nullptr;

    // Allocate memory on GPU
    cudaMalloc((void**)&a_gpu, n * sizeof(float));
    cudaMalloc((void**)&b_gpu, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(a_gpu, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    __multiply__<<<blocks, threadsPerBlock>>>(a_gpu, b_gpu, n);

    // Error check and sync
    cudaGetLastError();
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(b, b_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(a_gpu);
    cudaFree(b_gpu);
}
