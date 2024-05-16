#include <stdio.h>

#include "flash.h"

int main(void) {
    Matrix* m = InitMatrix(3, 3);
    double m_values[] = {8, -2, 5, 3, 0, 1, 3, 5, 6};
    SetElements(m, m_values);
    Matrix* x = InitMatrix(3, 3);
    double x_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    SetElements(x, x_values);

    PrintMatrix(m);
    printf("Rank of m: %d\n", MatrixRank(m));
    PrintMatrix(x);
    printf("Rank of x: %d\n", MatrixRank(x));

    FreeMatrix(m);
    FreeMatrix(x);
}
