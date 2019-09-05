#include <stdio.h>
#include <immintrin.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>


#define SIZE 1024
double b[SIZE][SIZE];
double a[SIZE][SIZE];

void func(double a[SIZE][SIZE]) {
    if(fork() == 0) {
        for(int j = 0; j < 64; j++) {
            for(int i = 0; i < SIZE; i++) {
                a[j][i] = 0;
                   
                for(int k = 0; k < SIZE; k++) {        
                    a[j][i] += b[i][k] + b[j][k];
                }

            }
        }

        printf("first: %f\n", a[0][0]);
        exit(0);
    }
}


void main(int argc, char **argv) {
    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) {
            b[i][j] = 20.19;
        }
    }

    for (int t = 0; t < 16; t++) {
        
        func(a + t * 64);
    }

    int some;
    for (;waitpid(-1, &some, 0) != -1;);
    
    for (int t = 0; t < 16; t++) {
        printf("second: %f\n", (a+t*64)[0][0]);
    }

    /*for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }*/



}
