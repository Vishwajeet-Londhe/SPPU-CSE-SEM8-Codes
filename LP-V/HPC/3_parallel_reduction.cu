#include <stdio.h>
#include <limits.h>

#define BLOCK_SIZE 256

__global__ void reduceSum(int *input, int *output, int n) {
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

__global__ void reduceMin(int *input, int *output, int n) {
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : INT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            if (sdata[tid + s] < sdata[tid])
                sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

__global__ void reduceMax(int *input, int *output, int n) {
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : INT_MIN;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main() {
    int n = 1024;
    int h_input[n];

    for (int i = 0; i < n; i++)
        h_input[i] = i + 1;

    int *d_input, *d_output;
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, gridSize * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int h_output[gridSize];

    // SUM
    reduceSum<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < gridSize; i++) sum += h_output[i];

    // MIN
    reduceMin<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int min = h_output[0];
    for (int i = 1; i < gridSize; i++)
        if (h_output[i] < min) min = h_output[i];

    // MAX
    reduceMax<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int max = h_output[0];
    for (int i = 1; i < gridSize; i++)
        if (h_output[i] > max) max = h_output[i];

    float avg = (float)sum / n;

    printf("Min = %d\n", min);
    printf("Max = %d\n", max);
    printf("Sum = %d\n", sum);
    printf("Average = %.2f\n", avg);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}