#include <stdio.h>
#include <immintrin.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

#define SIZE 1024
double b[SIZE][SIZE];

size_t ind(int x, int y) {
    return y * SIZE + x;
}


void func(double *a) {
    if(fork() == 0) {
        for(int j = 0; j < 64; j++) {
            for(int i = 0; i < SIZE; i++) {
                a[ind(i, j)] = 0;
                   
                for(int k = 0; k < SIZE; k++) {        
                    a[ind(i, j)] += b[i][k] + b[j][k];
                }

            }
        }

//        printf("first: %f\n", *a);
        exit(0);
    }
}


void main(int argc, char **argv) {
    
    double *a = mmap(NULL, SIZE*SIZE*sizeof(*a), PROT_WRITE, MAP_ANON | MAP_SHARED, -1, 0);

    if (a == MAP_FAILED) {
        printf("SHIT\n");

        exit(0);
    }

    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) {
            b[i][j] = 20.19;
        }
    }

    for (int t = 0; t < 16; t++) {
        
        func(a + t * 64 * SIZE);
    }
    
    int some;
    for (;waitpid(-1, &some, 0) != -1;);
    /*double check = a[0];    

    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) {
            if(a[ind(i, j)] != check) {
                printf("SHIT not working %lf %d %d\n", a[ind(i, j)], i, j);
            }
        }
    }*/

    munmap(a, sizeof(*a) * SIZE * SIZE);


}
