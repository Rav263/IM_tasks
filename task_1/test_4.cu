#include <stdio.h>

#define N  40

__global__ void MatAdd(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

size_t ind(int x, int y) {
    return y * N + x;
}

int main() {
    float A[N * N];
    float B[N * N];
    float C[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[ind(j, i)] = 3.4;
            B[ind(j, i)] = 5.4;
        }
    }

    float *a;
    float *b;
    float *c;

    cudaMalloc((void **) &a, N * N * sizeof(float));
    cudaMalloc((void **) &b, N * N * sizeof(float));
    cudaMalloc((void **) &c, N * N * sizeof(float));

    cudaMemcpy(a, A, sizeof(*a) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, sizeof(*a) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(c, C, sizeof(*a) * N * N, cudaMemcpyHostToDevice);

    int numBlocks = 1;
    dim3 threadsPerBlock(N * N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(a, b, c);

    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < N; j++) {
            printf("%f ", c[ind(j, i)]);
        }
        printf("\n");
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

