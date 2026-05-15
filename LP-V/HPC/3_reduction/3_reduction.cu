// Assignment 3: Parallel Reduction (Min, Max, Sum) using CUDA
// Compile: nvcc 3_reduction.cu -o 3_reduction
// Run:     ./3_reduction

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cuda_runtime.h>
using namespace std;

#define BLOCK 256

#define CHECK(c) { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        cout << "CUDA error: " << cudaGetErrorString(e) << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
}

__global__ void gpuSum(int *in, unsigned long long *out, int n) {
    __shared__ int s[BLOCK];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (tid < n) ? in[tid] : 0;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out, (unsigned long long)s[0]);
}

__global__ void gpuMin(int *in, int *out, int n) {
    __shared__ int s[BLOCK];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (tid < n) ? in[tid] : INT_MAX;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] = min(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMin(out, s[0]);
}

__global__ void gpuMax(int *in, int *out, int n) {
    __shared__ int s[BLOCK];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (tid < n) ? in[tid] : INT_MIN;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] = max(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMax(out, s[0]);
}

int main() {
    // Hardcoded sizes: small to large so chain graph lines cross
    int sizes[] = {10000, 100000, 1000000, 5000000, 20000000, 50000000};
    int n = 6;

    int smCount;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

    // Warm-up: force CUDA context init before any timed run
    int *tmp;
    cudaMalloc(&tmp, sizeof(int));
    gpuMax<<<1, 1>>>(tmp, tmp, 1);
    cudaDeviceSynchronize();
    cudaFree(tmp);

    cout << "Vishwajeet Londhe BE-B 41237" << endl << endl;
    cout << "Size\t\tSeq(s)\t\tPar(s)\t\tSpeedup\t\tEfficiency" << endl;

    ofstream csv("3_reduction.csv");
    csv << "size,seq_time,par_time,speedup,efficiency\n";

    for (int t = 0; t < n; t++) {
        int sz = sizes[t];

        int *data = new int[sz];
        srand(42);
        for (int i = 0; i < sz; i++)
            data[i] = rand() % 10000;

        // CPU sequential
        double cpuStart = (double)clock() / CLOCKS_PER_SEC;
        unsigned long long s = 0;
        int mn = INT_MAX, mx = INT_MIN;
        for (int i = 0; i < sz; i++) {
            s += data[i];
            if (data[i] < mn) mn = data[i];
            if (data[i] > mx) mx = data[i];
        }
        double seqTime = (double)clock() / CLOCKS_PER_SEC - cpuStart;

        // GPU parallel
        int *d_in, *d_min, *d_max;
        unsigned long long *d_sum;
        int h_min = INT_MAX, h_max = INT_MIN;

        CHECK(cudaMalloc(&d_in,  sz * sizeof(int)));
        CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
        CHECK(cudaMalloc(&d_min, sizeof(int)));
        CHECK(cudaMalloc(&d_max, sizeof(int)));
        CHECK(cudaMemcpy(d_in, data, sz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
        CHECK(cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice));

        int blocks = (sz + BLOCK - 1) / BLOCK;

        cudaEvent_t e1, e2;
        float ms;
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventRecord(e1);

        gpuSum<<<blocks, BLOCK>>>(d_in, d_sum, sz);
        gpuMin<<<blocks, BLOCK>>>(d_in, d_min, sz);
        gpuMax<<<blocks, BLOCK>>>(d_in, d_max, sz);

        cudaEventRecord(e2);
        cudaEventSynchronize(e2);
        cudaEventElapsedTime(&ms, e1, e2);
        double parTime = ms / 1000.0;

        double speedup = seqTime / parTime;
        double eff = speedup / smCount;

        cout << sz << "\t\t" << seqTime << "\t\t" << parTime << "\t\t"
             << speedup << "\t\t" << eff << endl;

        csv << sz << "," << seqTime << "," << parTime << "," << speedup << "," << eff << "\n";

        cudaFree(d_in);
        cudaFree(d_sum);
        cudaFree(d_min);
        cudaFree(d_max);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        delete[] data;
    }

    csv.close();
    cout << "\nSaved to 3_reduction.csv" << endl;
    return 0;
}
