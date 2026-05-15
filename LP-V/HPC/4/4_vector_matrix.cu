%%cuda

// Assignment 4: Vector Addition using CUDA
// Compile: nvcc vector_add.cu -o output
// Run:     ./output

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel
__global__ void vecAdd(float *a, float *b, float *c, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
        c[i] = a[i] + b[i];
}

// Sequential Vector Addition
void cpuVecAdd(float *a, float *b, float *c, int n) {

    for(int i=0;i<n;i++)
        c[i] = a[i] + b[i];
}

int main() {

    cout<<"Vector Addition using CUDA"<<endl<<endl;

    int sizes[] = {1000,100000,1000000,5000000,10000000};

    ofstream csv("vector_add.csv");

    csv<<"size,seq_time,par_time,speedup,efficiency\n";

    int smCount;

    cudaDeviceGetAttribute(&smCount,
                           cudaDevAttrMultiProcessorCount,
                           0);

    cout<<"Size\tSeq Time\tPar Time\tSpeedup\tEfficiency"<<endl;

    for(int t=0;t<5;t++) {

        int n = sizes[t];

        float *h_a = new float[n];
        float *h_b = new float[n];
        float *h_c = new float[n];

        for(int i=0;i<n;i++) {
            h_a[i] = rand()%100;
            h_b[i] = rand()%100;
        }

        // Sequential Timing
        double start = (double)clock()/CLOCKS_PER_SEC;

        cpuVecAdd(h_a,h_b,h_c,n);

        double seqTime =
        (double)clock()/CLOCKS_PER_SEC - start;

        // Device Memory
        float *d_a,*d_b,*d_c;

        cudaMalloc(&d_a,n*sizeof(float));
        cudaMalloc(&d_b,n*sizeof(float));
        cudaMalloc(&d_c,n*sizeof(float));

        cudaMemcpy(d_a,h_a,
                   n*sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_b,h_b,
                   n*sizeof(float),
                   cudaMemcpyHostToDevice);

        // CUDA Timing
        cudaEvent_t e1,e2;

        cudaEventCreate(&e1);
        cudaEventCreate(&e2);

        int threads = 256;
        int blocks = (n + threads - 1)/threads;

        cudaEventRecord(e1);

        vecAdd<<<blocks,threads>>>(d_a,d_b,d_c,n);

        cudaEventRecord(e2);

        cudaEventSynchronize(e2);

        float ms;

        cudaEventElapsedTime(&ms,e1,e2);

        double parTime = ms/1000.0;

        double speedup = seqTime/parTime;

        double efficiency = speedup/smCount;

        cout<<n<<"\t"
            <<seqTime<<"\t"
            <<parTime<<"\t"
            <<speedup<<"\t"
            <<efficiency<<endl;

        csv<<n<<","
           <<seqTime<<","
           <<parTime<<","
           <<speedup<<","
           <<efficiency<<"\n";

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        cudaEventDestroy(e1);
        cudaEventDestroy(e2);

        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
    }

    csv.close();

    cout<<"\nSaved to vector_add.csv"<<endl;

    return 0;
}