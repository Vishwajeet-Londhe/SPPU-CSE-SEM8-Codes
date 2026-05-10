#include <stdio.h>
#include <cuda.h>

#define SIZE 1024
#define THREADS 256

__global__ void sum(int *a,int *o){
    __shared__ int s[THREADS];
    int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;

    s[t]=a[i];
    __syncthreads();

    for(int st=blockDim.x/2;st>0;st/=2){
        if(t<st) s[t]+=s[t+st];
        __syncthreads();
    }

    if(t==0) o[blockIdx.x]=s[0];
}

__global__ void min(int *a,int *o){
    __shared__ int s[THREADS];
    int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;

    s[t]=a[i];
    __syncthreads();

    for(int st=blockDim.x/2;st>0;st/=2){
        if(t<st && s[t+st]<s[t]) s[t]=s[t+st];
        __syncthreads();
    }

    if(t==0) o[blockIdx.x]=s[0];
}

__global__ void max(int *a,int *o){
    __shared__ int s[THREADS];
    int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;

    s[t]=a[i];
    __syncthreads();

    for(int st=blockDim.x/2;st>0;st/=2){
        if(t<st && s[t+st]>s[t]) s[t]=s[t+st];
        __syncthreads();
    }

    if(t==0) o[blockIdx.x]=s[0];
}

int main(){

    int a[SIZE],out[SIZE/THREADS];

    for(int i=0;i<SIZE;i++) a[i]=i+1;

    int *da,*do_;
    int blocks=SIZE/THREADS;

    cudaMalloc(&da,SIZE*sizeof(int));
    cudaMalloc(&do_,blocks*sizeof(int));

    cudaMemcpy(da,a,SIZE*sizeof(int),cudaMemcpyHostToDevice);

    cudaEvent_t s,e;
    float t;

    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // SUM
    cudaEventRecord(s);
    sum<<<blocks,THREADS>>>(da,do_);
    cudaEventRecord(e);

    cudaEventSynchronize(e);
    cudaEventElapsedTime(&t,s,e);

    cudaMemcpy(out,do_,blocks*sizeof(int),cudaMemcpyDeviceToHost);

    int sm=0;

    for(int i=0;i<blocks;i++) sm+=out[i];

    printf("Sum=%d\nTime=%f ms\n\n",sm,t);

    // MIN
    cudaEventRecord(s);
    min<<<blocks,THREADS>>>(da,do_);
    cudaEventRecord(e);

    cudaEventSynchronize(e);
    cudaEventElapsedTime(&t,s,e);

    cudaMemcpy(out,do_,blocks*sizeof(int),cudaMemcpyDeviceToHost);

    int mn=out[0];

    for(int i=1;i<blocks;i++)
        if(out[i]<mn) mn=out[i];

    printf("Min=%d\nTime=%f ms\n\n",mn,t);

    // MAX
    cudaEventRecord(s);
    max<<<blocks,THREADS>>>(da,do_);
    cudaEventRecord(e);

    cudaEventSynchronize(e);
    cudaEventElapsedTime(&t,s,e);

    cudaMemcpy(out,do_,blocks*sizeof(int),cudaMemcpyDeviceToHost);

    int mx=out[0];

    for(int i=1;i<blocks;i++)
        if(out[i]>mx) mx=out[i];

    printf("Max=%d\nTime=%f ms\n\n",mx,t);

    // AVG
    cudaEventRecord(s);

    float avg=(float)sm/SIZE;

    cudaEventRecord(e);

    cudaEventSynchronize(e);
    cudaEventElapsedTime(&t,s,e);

    printf("Average=%.2f\nTime=%f ms\n",avg,t);

    cudaFree(da);
    cudaFree(do_);
}