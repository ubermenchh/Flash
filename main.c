#include <stdio.h>

#include "flash.h"

int main(void) {
    Vector* v = InitVector(2);
    Vector* w = InitVector(3);
    Vector* z = zeros_vector(5);
    Vector* o = ones_vector(5);
    Vector* r = random_vector(6, 0);
    double v_elem[] = {2.3, 3.4};
    double w_elem[] = {3.4, 5.6, 4.5};
    VectorSetElements(v, v_elem);
    VectorSetElements(w, w_elem);
    PrintVector(v);
    PrintVector(w);
    PrintVector(z);
    PrintVector(o);
    PrintVector(r);

    FreeVector(v);
    FreeVector(w);
    FreeVector(z);
    FreeVector(o);
    FreeVector(r);
    /*
    int x = 10;
    Matrix* m = InitMatrix(3, 3);
    Matrix* n = InitMatrix(3, 3);
    Matrix* p = InitMatrix(4, 4);
    Matrix* r = RandMatrix(3, 3, 0);
    double m_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double n_values[] = {-3, -4, -2, 0, 1, 5, 8, 2, -5};
    double p_values[] = {1, 3, -8, 2, 4, -9, 0, 2, 1, 1, 5, -4, -5, -7, 8, 1};
    SetElements(m, m_values);
    SetElements(n, n_values);
    SetElements(p, p_values);
    double det_p = determinant(p);
    double trace_p = trace(p);
    double frob_norm = frobenius_norm(p);
    double lone_norm = l1_norm(p);
    double inf_norm = infinity_norm(p);
    Matrix* a = concat(m, n, 0);
    Matrix* i = copy(p);
    //printf("Matrix m: \n");
    //PrintMatrix(m);
    printf("Matrix p: \n");
    PrintMatrix(i);
    printf("Matrix i: \n");
    PrintMatrix(i);
    printf("Matrix a: \n");
    PrintMatrix(a);

    //printf("Matrix det_p: %f\n", det_p);
    //printf("Frobenius Norm: %f | L1 Norm: %f | Infinity Norm: %f \n", norm(m, "frobenius"), norm(m, "l1"), norm(m, "infinity"));

    FreeMatrix(m);
    FreeMatrix(n);
    FreeMatrix(r);
    FreeMatrix(a);
    FreeMatrix(p);
    FreeMatrix(i);
    */
}
