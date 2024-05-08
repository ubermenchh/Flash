#include <stdio.h>

#include "flash.h"

int main(void) {
    Vector2 v;
    v.x = 3.4;
    v.y = 4.5;

    Vector2 w;
    w.x = 2.1;
    w.y = 1.2;

    int x = 2;

    Vector2 sum = add(v, w);
    Vector2 sub = subtract(v, w);
    Vector2 double_v = multiply_scalar(v, x);
    Vector2 double_w = multiply_scalar(w, x);
    Vector2 half_v = divide_scalar(v, x);
    Vector2 half_w = divide_scalar(w, x);
    float magnitude_v = magnitude(v);
    float magnitude_w = magnitude(w);
    float vdotw = dot_product(v, w);
    float angle = Vector2_angle(v, w);
    float vcrossw = cross_product(v, w);

    print_vector2(v);
    print_vector2(w);
    print_vector2(sum);
    print_vector2(sub);
    print_vector2(double_v);
    print_vector2(double_w);
    print_vector2(half_v);
    print_vector2(half_w);
    printf("magnitude of v: %f\n", magnitude_v);
    printf("magnitude of w: %f\n", magnitude_w);
    printf("v . w: %f\n", vdotw);
    printf("Angle between v and w: %f radians\n", angle);
    printf("Angle between v and w: %f degrees\n", radian_to_degrees(angle));
    printf("v x w: %f\n", vcrossw);
}
