#include "flash.h"

int main(void) {
    Matrix* m = InitMatrix(3, 3);
    double m_vals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    SetElements(m, m_vals);
    Matrix* n = InitMatrix(3, 3);
    double n_vals[] = {1, 2, 4, 6, 5, 8, 1, 8, 9};
    SetElements(n, n_vals);

    PrintMatrix(m);
    PrintMatrix(n);
    PrintMatrix(MatrixEq(m, n));

    FreeMatrix(m);
    FreeMatrix(n);
    
    return 0;
}

