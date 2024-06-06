#include <stdio.h>
#include <time.h>

#include "flash.h"

int main(void) {
    Matrix* m = InitMatrix(3, 3);
    Matrix* n = InitMatrix(3, 1);
    double m_vals[] = {
        1, 2, 3,
        4, 9, 6,
        7, 10, 9
    };
    double n_vals[] = { 1, 2, 3 };
    SetElements(m, m_vals);
    SetElements(n, n_vals);

    PrintMatrix(m);
    PrintMatrix(n);
    PrintMatrix(MatrixAdd(m, m));

    FreeMatrix(m);
    FreeMatrix(n);
}

