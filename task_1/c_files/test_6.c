#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK

#define SIZE 1024
double c[SIZE];
double a[SIZE * SIZE];
double b[SIZE * SIZE];

void main( int argc, char **argv) {

    for (int i = 0; i < SIZE * SIZE; i++) { 
        b[i] = 20.19;
    }
  
    for (int i = 0; i < SIZE * SIZE; i++) {
        c[i / SIZE] += b[i];
    }


    for (int i = 0; i < SIZE * SIZE; i++) {
        a[i] = c[i / SIZE] + c[i % SIZE];
    }

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
}
