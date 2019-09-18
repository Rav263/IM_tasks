#include <stdio.h>

#define SIZE 512

int main() {
    float vec1[SIZE];
    float vec2[SIZE];
    float vec3[SIZE];

    for (int i = 0; i < SIZE; i++) {
        vec1[i] = i;
        vec2[i] = i;
    }


    for (int i = 0; i < SIZE; i++) {
        vec3[i] = vec1[i] + vec2[i];
    }

    for (int i = 0; i < SIZE; i++) {
        printf("Element #%i: %.1f\n", i, vec3[i]);
    }
}
