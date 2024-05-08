#include "flash.h"

Vector2 add(Vector2 v, Vector2 w) {
    return (Vector2){v.x + w.x, v.y + w.y};
}

Vector2 subtract(Vector2 v, Vector2 w) {
    return (Vector2){v.x - w.x, v.y - w.y};
}

Vector2 multiply_scalar(Vector2 v, int x) {
    return (Vector2){v.x * x, v.y * x};
}

Vector2 divide_scalar(Vector2 v, int x) {
    return (Vector2){v.x / x, v.y / x};
}

void print_vector2(Vector2 v) {
    printf("[%f %f]\n", v.x, v.y);
}

float magnitude(Vector2 v) {
    return sqrt((v.x*v.x) + (v.y*v.y));
}

float dot_product(Vector2 v, Vector2 w) {
    return (v.x * w.x) + (v.y * w.y); 
}

float radian_to_degrees(float x) {
    return (x * 180) / M_PI;
}

float Vector2_angle(Vector2 v, Vector2 w) {
    float mag_v = magnitude(v);
    float mag_w = magnitude(w);
    float vdotw = dot_product(v, w);

    float y = vdotw / (mag_v * mag_w);
    return acos(y);
}

float cross_product(Vector2 v, Vector2 w) {
    float mag_v = magnitude(v);
    float mag_w = magnitude(w);
    float angle = radian_to_degrees(Vector2_angle(v, w));

    return (mag_v * mag_w) * sin(angle);
}
