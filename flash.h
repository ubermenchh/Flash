#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define MAX_ROWS 1024
#define MAX_COLS 1024

typedef struct {
    size_t size;
    double* data;
} Vector;

typedef struct {
    int rows; 
    int cols;
    double* data;
} Matrix;

double radian_to_degrees(double x);

Vector* InitVector(size_t size);
void FreeVector(Vector* v);
void VectorSetElements(Vector* v, double* values);
void vector_set(Vector* v, size_t index, double value);
double vector_get(Vector* v, size_t index);
void PrintVector(Vector* v);
Vector* vector_add(Vector* v, Vector* w);
Vector* vector_sub(Vector* v, Vector* w);
Vector* vector_scale(Vector* v, int x);
double vector_norm(Vector* v);
double vector_dotproduct(Vector* v, Vector* w);
double vector_angle(Vector* v, Vector* w);
double vector_crossproduct(Vector* v, Vector* w);
bool vector_equal(Vector* v, Vector* w);
Vector* vector_normalize(Vector* v);
Vector* zeros_vector(size_t size);
Vector* ones_vector(size_t size);
Vector* random_vector(size_t size, int seed);
Vector* vector_copy(Vector* v);
Vector* vector_multiply(Vector* v, Vector* w);
double vector_projection(Vector* v, Vector* w);


Matrix* InitMatrix(int rows, int cols);
void FreeMatrix(Matrix* m);
void SetElements(Matrix* m, double* values);
void PrintMatrix(Matrix* m);
Matrix* RandMatrix(int rows, int cols, int seed);
Matrix* matadd(Matrix* m, Matrix* n);
Matrix* matsub(Matrix* m, Matrix* n);
Matrix* scalarmul(Matrix* m, int x);
Matrix* transpose(Matrix* m);
Matrix* ones(int rows, int cols);
Matrix* zeros(int rows, int cols);
Matrix* identity(int side);
Matrix* matmul(Matrix* m, Matrix* n);
Matrix* slice(Matrix* m, int from_rows, int to_rows, int from_cols, int to_cols);
void LUDecomp(Matrix* A, Matrix** L, Matrix** U);
double determinant(Matrix* m);
double trace(Matrix* m);
double frobenius_norm(Matrix* m);
double l1_norm(Matrix* m);
double infinity_norm(Matrix* m);
double norm(Matrix* m, char* type);
Matrix* concat(Matrix* m, Matrix* n, int axis);
Matrix* copy(Matrix* m);
