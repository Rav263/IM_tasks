#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

#define CHECK

#define SIZE 10000
#define THREADS 4

//double c[SIZE];
double b[SIZE * SIZE];
double a[SIZE * SIZE];
double c[SIZE];

void fork_func_1(size_t start_index, size_t end_index) {
    for (int i = start_index; i < end_index; i++) {
        a[i] = c[i / SIZE] + c[i % SIZE];
    }
}

void fork_func_0(size_t start_index, size_t end_index) {
    for (int i = start_index; i < end_index; i++) {
        c[i / SIZE] += b[i];
    }
}

int main() {
    int block_size = SIZE * SIZE / THREADS;
    for (int i = 0; i < SIZE * SIZE; i++) { 
        b[i] = 20.19;
    }

    std::vector<std::thread> threads;

    for (int i = 0; i < THREADS; i++) {
        threads.push_back(std::thread(fork_func_0, i * block_size, 
                         (i + 1) * block_size));
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i] = std::thread(fork_func_1, i * block_size, 
                                (i + 1) * block_size);
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
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
    //munmap(a, sizeof(*a) * SIZE * SIZE);
    //munmap(c, sizeof(*a) * SIZE);
}
