#include <stdio.h>

#include "flash.h"

int main(void) {
    Matrix* m = InitMatrix(3, 3);
    double m_values[] = {10, -45, 92, -91, 21, 4, 23, 0, -81};
    SetElements(m, m_values);

    PrintMatrix(m);
    printf("%f | %f \n", MatrixDeterminant(m), MatrixLogDeterminant(m));

    FreeMatrix(m);
}

