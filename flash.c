#include "flash.h"

/*
****************************
***** HELPER FUNCTIONS *****
****************************
*/
float radian_to_degrees(float x) {
    /* Converts radian to degree */
    return (x * 180) / M_PI;
}

/*
*******************************
***** 2D VECTOR FUNCTIONS *****
*******************************
*/
Vector2d Add2d(Vector2d v, Vector2d w) {
    /* Adds two 2d vectors element-wise */ 
    return (Vector2d){v.x + w.x, v.y + w.y};
}

Vector2d Subtract2d(Vector2d v, Vector2d w) {
    /* Substracts two 2d vectors element-wise */
    return (Vector2d){v.x - w.x, v.y - w.y};
}

Vector2d Scale2d(Vector2d v, int x) {
    /* Multiplies an int to each component of a 2d vector */
    return (Vector2d){v.x * x, v.y * x};
}

Vector2d DivideScalar2d(Vector2d v, int x) {
    /* Divides each component of a 2d vector by an int */
    return (Vector2d){v.x / x, v.y / x};
}

void Print2d(Vector2d v) {
    /* Prints a 2d vector: [x, y, x] */
    printf("[%f %f]\n", v.x, v.y);
}

float Norm2d(Vector2d v) {
    /* Calculates the magnitude of a 2d Vector */
    return sqrt((v.x*v.x) + (v.y*v.y));
}

float DotProduct2d(Vector2d v, Vector2d w) {
    /* Dot Product between two 2d vectors */
    return (v.x * w.x) + (v.y * w.y); 
}

float Angle2d(Vector2d v, Vector2d w) {
    /* Calculates angle between two 2d vector */
    float mag_v = Norm2d(v);
    float mag_w = Norm2d(w);
    float vdotw = DotProduct2d(v, w);

    float y = vdotw / (mag_v * mag_w);
    return acos(y);
}

float CrossProduct2d(Vector2d v, Vector2d w) {
    /* Calculates the cross product of two 2d vectors */
    float mag_v = Norm2d(v);
    float mag_w = Norm2d(w);
    float angle = radian_to_degrees(Angle2d(v, w));

    return (mag_v * mag_w) * sin(angle);
}

bool Equal2d(Vector2d v, Vector2d w) {
    /* Check whether two 2d vectors are equal */
    return (v.x == w.x) && (v.y == w.y); 
}

Vector2d Normalize2d(Vector2d v) {
    /* Normalized a 2d Vector */
    float norm_v = Norm2d(v);
    return (Vector2d){v.x / norm_v, v.y / norm_v};
}

Vector2d Zeros2d(void) {
    return (Vector2d){0.0, 0.0};
}

Vector2d Ones2d(void) {
    return (Vector2d){1.0, 1.0};
}

Vector2d Init2d(int seed) {
    if (seed != 0) {
        srand(seed);
    } else {
        srand(time(NULL));
    }
    return (Vector2d){(float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX};
}

Vector2d Copy2d(Vector2d v) {
    return (Vector2d){v.x, v.y};
}

Vector2d Multiply2d(Vector2d v, Vector2d w){
    return (Vector2d){v.x * w.x, v.y * w.y};
}

/*
*******************************
***** 3D VECTOR FUNCTIONS *****
*******************************
*/
Vector3d Add3d(Vector3d v, Vector3d w) {
    return (Vector3d){v.x + w.x, v.y + w.y, v.z + w.z};
}

Vector3d Subtract3d(Vector3d v, Vector3d w) {
    return (Vector3d){v.x - w.x, v.y - w.y, v.z + w.z};
}

Vector3d Scale3d(Vector3d v, int x) {
    return (Vector3d){v.x * x, v.y * x, v.z * x};
}

Vector3d DivideScalar3d(Vector3d v, int x) {
    return (Vector3d){v.x / x, v.y / x, v.z / x};
}

void Print3d(Vector3d v) {
    printf("[%f %f %f]", v.x, v.y, v.z);
}

float Norm3d(Vector3d v) {
    return sqrt((v.x*v.x) + (v.y*v.y) + (v.z*v.z));
}

float DotProduct3d(Vector3d v, Vector3d w) {
    return (v.x * w.x) + (v.y * w.y) + (v.z * w.z);
}

float Angle3d(Vector3d v, Vector3d w) {
    float mag_v = Norm3d(v);
    float mag_w = Norm3d(w);
    float vdotw = DotProduct3d(v, w);

    float y = vdotw / (mag_v * mag_w);
    return acos(y);
}

float CrossProduct3d(Vector3d v, Vector3d w) {
    float mag_v = Norm3d(v);
    float mag_w = Norm3d(w);
    float angle = radian_to_degrees(Angle3d(v, w));

    return (mag_v * mag_w) * sin(angle);
}

bool Equal3d(Vector3d v, Vector3d w) {
    return (v.x == w.x) && (v.y == w.y) && (v.z == w.z); 
}

Vector3d Normalize3d(Vector3d v) {
    float norm_v = Norm3d(v);
    return (Vector3d){v.x / norm_v, v.y / norm_v, v.z / norm_v};
}

Vector3d Zeros3d(void) {
    return (Vector3d){0.0, 0.0, 0.0};
}

Vector3d Ones3d(void) {
    return (Vector3d){1.0, 1.0, 1.0};
}

Vector3d Init3d(int seed) {
    if (seed != 0) {
        srand(seed);
    } else {
        srand(time(NULL));
    }
    return (Vector3d){(float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX};
}

Vector3d Copy3d(Vector3d v) {
    return (Vector3d){v.x, v.y, v.z};
}

Vector3d Multiply3d(Vector3d v, Vector3d w){
    return (Vector3d){v.x * w.x, v.y * w.y, v.z * w.z};
}

