#include <stdio.h>

#define N 4   // Matrix size (you can increase)

__global__ void matrixMul(int A[N][N], int B[N][N], int C[N][N]) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;

    for (int k = 0; k < N; k++) {
        sum += A[row][k] * B[k][col];
    }

    C[row][col] = sum;
}

int main() {
    int A[N][N], B[N][N], C[N][N];

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i * j;
        }
    }

    int (*d_A)[N], (*d_B)[N], (*d_C)[N];

    cudaMalloc((void**)&d_A, sizeof(int) * N * N);
    cudaMalloc((void**)&d_B, sizeof(int) * N * N);
    cudaMalloc((void**)&d_C, sizeof(int) * N * N);

    cudaMemcpy(d_A, A, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * N * N, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);

    matrixMul<<<1, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

    printf("Result Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}