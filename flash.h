#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>

typedef enum {
    VECTOR_2D,
    VECTOR_3D,
} VectorType;

typedef struct {
    double x, y;
} Vector2d;

typedef struct {
    double x, y, z;
} Vector3d;

float radian_to_degrees(float x);

Vector2d Add2d(Vector2d v, Vector2d w);
Vector2d Subtract2d(Vector2d v, Vector2d w);
Vector2d Scale2d(Vector2d v, int x);
Vector2d DivideScalar2d(Vector2d v, int x);
void Print2d(Vector2d v);
float Norm2d(Vector2d v);
float DotProduct2d(Vector2d v, Vector2d w);
float Angle2d(Vector2d v, Vector2d w);
float CrossProduct2d(Vector2d v, Vector2d w);
bool Equal2d(Vector2d v, Vector2d w);
Vector2d Normalize2d(Vector2d v);
Vector2d Zeros2d(void);
Vector2d Ones2d(void);
Vector2d Init2d(int seed);
Vector2d Copy2d(Vector2d v);
Vector2d Multiply2d(Vector2d v, Vector2d w);

Vector3d Add3d(Vector3d v, Vector3d w);
Vector3d Subtract3d(Vector3d v, Vector3d w);
Vector3d Scale3d(Vector3d v, int x);
Vector3d DivideScalar3d(Vector3d v, int x);
void Print3d(Vector3d v);
float Norm3d(Vector3d v);
float DotProduct3d(Vector3d v, Vector3d w);
float Angle3d(Vector3d v, Vector3d w);
float CrossProduct3d(Vector3d v, Vector3d w);
bool Equal3d(Vector3d v, Vector3d w);
Vector3d Normalize3d(Vector3d v);
Vector3d Zeros3d(void);
Vector3d Ones3d(void);
Vector3d Init3d(int seed);
Vector3d Copy3d(Vector3d v);
Vector3d Multiply3d(Vector3d v, Vector3d w);

