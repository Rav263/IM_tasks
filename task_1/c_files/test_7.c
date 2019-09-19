#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>

#define CHECK

#define SIZE 1024
#define THREADS 4

//double c[SIZE];
double b[SIZE * SIZE];
double *a;
double *c;

void fork_func_1(size_t start_index, size_t end_index) {
    if (fork() == 0) {
        for (int i = start_index; i < end_index; i++) {
            a[i] = c[i / SIZE] + c[i % SIZE];
        }
        exit(0);
    }
}

void fork_func_0(size_t start_index, size_t end_index) {
    if (fork() == 0) {
        for (int i = start_index; i < end_index; i++) {
            c[i / SIZE] += b[i];
        }
        exit(0);
    }
}

void main( int argc, char **argv) {
    a = mmap(NULL, SIZE * SIZE * sizeof(*a), PROT_WRITE, MAP_ANON | MAP_SHARED, -1, 0);
    c = mmap(NULL, SIZE * sizeof(*a), PROT_WRITE, MAP_ANON | MAP_SHARED, -1, 0);

    if (a == MAP_FAILED || c == MAP_FAILED) {
        printf("MAP ERROR\n");
        exit(0);
    }

    int block_size = SIZE * SIZE / THREADS;
    for (int i = 0; i < SIZE * SIZE; i++) { 
        b[i] = 20.19;
    }
    
    for (int i = 0; i < THREADS; i++) {
        fork_func_0(i * block_size, (i + 1) * block_size);
    }
    for(;wait(0) != -1;);
    
    for (int i = 0; i < THREADS; i++) {
        fork_func_1(i * block_size, (i + 1) * block_size);
    }
    for(;wait(0) != -1;);
#ifdef CHECK
    double temp = a[0];

    for (int i = 0; i < SIZE * SIZE; i++) {
        if (a[i] - temp < 0 ? temp - a[i] : a[i] - temp > 0.000000000001) {
            printf("НУ ДА НУ ДА, ПОШЁЛ Я НАХЕР\n");
            break;
        }
    }

    printf("%lf\n", a[0]);
#endif
    munmap(a, sizeof(*a) * SIZE * SIZE);
    munmap(c, sizeof(*a) * SIZE);
}
