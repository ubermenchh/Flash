#include <stdio.h>

#include "flash.h"

int main(void) {
    Matrix* m = InitMatrix(3, 4);
    double m_values[] = {8, -2, 5, 3, 0, 1, 3, 5, 6, -2, 3, 1};
    SetElements(m, m_values);
    Matrix* m_max = MatrixStdVals(m, 1);

    PrintMatrix(m);
    printf("Max: %f | Min: %f | Mean: %f | Std: %f |\n", MatrixMax(m), MatrixMin(m), MatrixMean(m), MatrixStd(m));
    PrintMatrix(m_max);

    FreeMatrix(m);
    FreeMatrix(m_max);
}
