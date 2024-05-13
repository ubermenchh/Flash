#include <stdio.h>

#include "flash.h"

int main(void) {
    Matrix* m = InitMatrix(3, 3);
    double m_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    SetElements(m, m_values);
    MatrixTuple LUDec = LUDecomposition(m);
    Matrix* prod = MatrixMul(LUDec.first, LUDec.second);

    PrintMatrix(m);
    PrintMatrix(LUDec.first);
    PrintMatrix(LUDec.second);
    PrintMatrix(prod);

    FreeMatrix(m);
    FreeMatrix(prod);
    FreeMatrixTuple(LUDec);
}
