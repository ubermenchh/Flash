#include <stdio.h>
#include <math.h>

typedef struct {
    float x;
    float y;
} Vector2;

Vector2 add(Vector2 v, Vector2 w);
Vector2 subtract(Vector2 v, Vector2 w);
Vector2 multiply_scalar(Vector2 v, int x);
Vector2 divide_scalar(Vector2 v, int x);
void print_vector2(Vector2 v);
float magnitude(Vector2 v);
float dot_product(Vector2 v, Vector2 w);
float radian_to_degrees(float x);
float Vector2_angle(Vector2 v, Vector2 w);
float cross_product(Vector2 v, Vector2 w);
