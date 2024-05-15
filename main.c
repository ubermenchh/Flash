#include <stdio.h>

#include "flash.h"

int main(void) {
    Matrix* m = InitMatrix(3, 3);
    double m_values[] = {8, -2, 5, 0, -1, -4, 9, 12, -10};
    SetElements(m, m_values);
    Matrix* x = InitMatrix(3, 3);
    double x_values[] = {1, 2, 3, 1, 4, 7, 9, 0, 9};
    SetElements(x, x_values);
    MatrixTuple qr = QRDecomposition(x);
    Matrix* y = MatrixMul(qr.first, qr.second);

    PrintMatrix(x);
    //matrix_to_identity(m);
    PrintMatrix(qr.first);
    PrintMatrix(qr.second);
    PrintMatrix(y);

    FreeMatrix(m);
    FreeMatrix(x);
    FreeMatrix(y);
    FreeMatrixTuple(qr);
}
