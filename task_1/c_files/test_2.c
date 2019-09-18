#include <stdio.h>
#include <immintrin.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>


#define SIZE 1024
double a[SIZE][SIZE];
double b[SIZE][SIZE];

void func(double *a[SIZE], double *b[SIZE], int start, int i, int j) {
    if(fork() == 0) {
        for (int k = 0; k < 64; k++) {
            a[i][j] += b[i][k + start] + b[j][k + start];
        }

        exit(0);
    }
}


void main(int argc, char **argv) {
  
    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) {
            b[i][j] = 20.19;
        }
    }


    for(int j = 0; j < SIZE; j++) {
        printf("fork: %d\n", j);
        for(int i = 0; i < SIZE; i++) {
            a[i][j] = 0;
            
                       
            for(int k = 0; k < 16; k++) {        
                func(a, b, k * 64, i, j);
            }

        }
    }
}
