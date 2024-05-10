#include <stdio.h>

#include "flash.h"

int main(void) {
    /*
    biVector2d v = {1.2, 3.4};
    Vector2d w = {4.5, 5.6};
    Vector3d a = {5, 4, 1};
    Vector3d b = {3, 2, 4};
    int x = 2;

    //Matrix m = {2, 2, {{2, 3}, {3, 4}}};

    Vector2d sum_of_vw = Add2d(v, w);
    Vector2d sub_of_vw = Subtract2d(v, w);
    Vector2d double_v = Scale2d(v, x);
    Vector2d double_w = Scale2d(w, x);
    Vector2d half_v = DivideScalar2d(v, x);
    Vector2d half_w = DivideScalar2d(w, x);
    float norm_v = Norm2d(v);
    float norm_w = Norm2d(w);
    float vdotw = DotProduct2d(v, w);
    float angle = Angle2d(v, w);
    float vcrossw = CrossProduct2d(v, w);
    Vector2d zero_vector = Zeros2d();
    Vector2d one_vector = Ones2d();
    Vector2d init_vector = Init2d(0);
    Vector2d vmulw = Multiply2d(v, w);
    float aprojb = Projection3d(a, b); 
    float adotb = DotProduct3d(a, b);
    float norm_a = Norm3d(a);
    float norm_b = Norm3d(b);

    printf("v:\t");
    Print2d(v);
    printf("w:\t");
    Print2d(w);
    printf("v + w:\t");
    Print2d(sum_of_vw);
    printf("v - w:\t");
    Print2d(sub_of_vw);
    printf("Double of v (2*v):\t");
    Print2d(double_v);
    printf("Double of w (2*w):\t");
    Print2d(double_v);
    printf("Half of v (v / 2):\t");
    Print2d(half_v);
    printf("Half of w (w / 2):\t");
    Print2d(half_w);
    printf("norm of v:\t%f\n", norm_v);
    printf("norm of w:\t%f\n", norm_w);
    printf("v . w:\t%f\n", vdotw);
    printf("Angle between v and w:\t%f radians\n", angle);
    printf("Angle between v and w:\t%f degrees\n", radian_to_degrees(angle));
    printf("v x w:\t%f\n", vcrossw);
    printf("v == w:\t%s\n", Equal2d(v, w) ? "true" : "false"); 
    printf("v == v:\t%s\n", Equal2d(v, v) ? "true" : "false"); 
    printf("normalized v:\t");
    Print2d(Normalize2d(v));
    printf("normalized w:\t");
    Print2d(Normalize2d(w));
    printf("Zeros Vector2d:\t");
    Print2d(zero_vector);
    printf("Ones Vector2d:\t");
    Print2d(one_vector);
    printf("Randomly Initialized Vector2d:\t");
    Print2d(init_vector);
    printf("v * w (element-wise):\t");
    Print2d(vmulw);

    printf("a . b: %f\n", adotb);
    printf("norm of a: %f | norm of b: %f\n", norm_a, norm_b);
    prbin/intf("Projection of a on b: %f\n", aprojb);
    */

    Matrix* m = InitMatrix(2, 2);
    Matrix* n = InitMatrix(2, 2);
    Matrix* r = RandMatrix(4, 4, 69); 
    double m_values[] = {1, 2, 3, 4};
    double n_values[] = {9, 8, 7, 6};
    SetElements(m, m_values);
    SetElements(n, n_values);
    printf("Matrix m: \n");
    PrintMatrix(m);
    printf("Matrix n: \n");
    PrintMatrix(n);
    printf("Matrix r: \n");
    PrintMatrix(r);

    FreeMatrix(m);
    FreeMatrix(n);
    FreeMatrix(r);
}
