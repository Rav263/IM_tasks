#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define SIZE 1024
#define THREADS 1024
#define BLOCKS SIZE / THREADS

#define CHECK

double a[SIZE][SIZE];
double b[SIZE][SIZE];
double c[SIZE];


__global__ void sum_matrix_lines(double *matrix, double *vec) {
    int y = (blockIdx.y * BLOCKS) + threadIdx.y;
    for (int x = 0; x < SIZE; x++) {
        vec[y] += matrix[y * SIZE + x];
    }
}

__global__ void gen_matrix_from_lines(double *matrix, double *vec) {
    int y = (blockIdx.y * BLOCKS) + threadIdx.y;
    
    for (int x = 0; x < SIZE; x++) {
        matrix[y * SIZE + x] = vec[x] + vec[y]; 
    }
}

int main() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            b[i][j] = 20.19;
        }
    }

    cudaDeviceReset();

    double *matrix_1;
    double *matrix_2;
    double *matrix_sum;

    cudaMalloc(&matrix_1,   SIZE * SIZE * sizeof(*matrix_1));
    cudaMalloc(&matrix_2,   SIZE * SIZE * sizeof(*matrix_2));
    cudaMalloc(&matrix_sum, SIZE *        sizeof(*matrix_sum));

    cudaMemcpy(matrix_2, b, SIZE * SIZE * sizeof(*matrix_2), cudaMemcpyHostToDevice);

    dim3 grid_size = {1, BLOCKS, 1};
    dim3 block_size = {1, THREADS, 1};

    sum_matrix_lines<<<block_size, grid_size, 0, 0>>>(matrix_2, matrix_sum);
    
    cudaDeviceSynchronize();

    gen_matrix_from_lines<<<block_size, grid_size, 0, 0>>>(matrix_1, matrix_sum);
    cudaDeviceSynchronize();

    cudaMemcpy(a, matrix_1, SIZE * SIZE * sizeof(*c), cudaMemcpyDeviceToHost);
    
    #ifdef CHECK
    double temp = a[0][0];

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (a[i][j] - temp < 0 ? temp - a[i][j] : a[i][j] - temp > 0.000000000001) {
                printf("НУ ДА НУ ДА, ПОШЁЛ Я НАХЕР\n");
                break;
            }
        }
    }

    printf("%lf\n", a[0][0]);
#endif

    cudaFree(matrix_1);
    cudaFree(matrix_2);
    cudaFree(matrix_sum);
}
