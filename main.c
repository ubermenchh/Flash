#include <stdio.h>
#include <time.h>

#include "flash.h"

int main(void) {
    clock_t start = clock();
    Matrix* m = RandMatrix(1000, 1000, 69);
    Matrix* n = RandMatrix(1000, 1000, 42);
    Matrix* prod = MatrixMul(m, n);
    clock_t end = clock();

    PrintMatrix(prod);
    double elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Runtime: %.3f milliseconds\n", elapsed_time);

    FreeMatrix(m);
    FreeMatrix(n);
    FreeMatrix(prod);
}

