#include <stdio.h>
#include <time.h>

#include "flash.h"

int main(void) {
    Matrix* m = RandMatrix(10, 10, 69);
    //double m_vals[] = {
    //    1, 2, 3,
    //    4, 9, 6,
    //    7, 10, 9
    //};
    //SetElements(m, m_vals);

    PrintMatrix(m);
    PrintVector(MatrixDiagonal(m, -3));

    FreeMatrix(m);
}

