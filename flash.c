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
    /* Returns a 2d vector of 0s -> [0.0 0.0] */
    return (Vector2d){0.0, 0.0};
}

Vector2d Ones2d(void) {
    /* Returns a 2d vector of 1s -> [1.0 1.0] */
    return (Vector2d){1.0, 1.0};
}

Vector2d Init2d(int seed) {
    /* Initializes a 2d vector randomly */
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

float Projection2d(Vector2d v, Vector2d w) {
    float vdotw = DotProduct2d(v, w);
    float mag_w = Norm2d(w);
    return vdotw / mag_w;
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

float Projection3d(Vector3d v, Vector3d w) {
    float vdotw = DotProduct3d(v, w);
    float mag_w = Norm3d(w);
    return vdotw / mag_w;
}

/*
****************************
**********MATRICES**********
****************************
*/

Matrix* InitMatrix(int rows, int cols) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (matrix == NULL) return NULL;

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->data = (double*)calloc(rows * cols, sizeof(double));
    if (matrix->data == NULL) {
        free(matrix);
        return NULL;
    }
    return matrix;
}

void FreeMatrix(Matrix* m) {
    free(m->data);
    free(m);
}
/*
Matrix* RandMatrix(int rows, int cols) {
    Matrix* m = InitMatrix(rows, cols);
    srand(time(NULL));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m->data[i][j] = (double)rand() / (double)RAND_MAX; 
        }
    }
}
*/ 

void SetElements(Matrix* m, double* values) {
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        m->data[i] = values[i];
    }
}

void PrintMatrix(Matrix* m) {
    int max_digits = 0;
    double max_val = 0.0;
    double min_val = 0.0;

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double val = fabs(m->data[i * m->cols + j]);
            if (val > max_val) {
                max_val = val;
            }
            if (val < min_val || (min_val == 0.0)) {
                min_val = val;
            }
        }
    }

    if ((max_val == 0.0 && min_val == 0.0) || (max_val == 1.0 && min_val == 1.0)) {
        max_digits = 1;
    } else {
        max_digits = (int)log10(max_val) + 1 + 8;
    }

    printf("[");
    for (int i = 0; i < m->rows; i++) {
        if (i == 0) {
            printf("[");
        } else {
            printf(" [");
        }
        for (int j = 0; j < m->cols; j++) {
            printf("%*.*f ", max_digits, 5, m->data[i * m->cols + j]);
        }
        if (i != m->rows-1) {
            printf(" ]\n");
        } else {
            printf(" ]");
        }
    }
    printf("]\n");
}

Matrix* RandMatrix(int rows, int cols, int seed) {
    if (seed != 0) {
        srand(0);
    } else {
        srand(time(NULL));
    }

    Matrix* m = InitMatrix(rows, cols);
    int size = rows * cols;
    double rand_array[size];
    for (int i = 0; i < size; i++) {
        rand_array[i] = (double)rand() / (double)RAND_MAX;
    }
    SetElements(m, rand_array);
    return m;
}

Matrix* matadd(Matrix* m, Matrix* n) {
    assert(m->rows * m->cols == n->rows * n->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[i * m->cols + j] = m->data[i * m->cols + j] + n->data[i * n->cols + j];
        }
    }
    return out;
}

Matrix* matsub(Matrix* m, Matrix* n) {
    assert(m->rows * m->cols == n->rows * n->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[i * m->cols + j] = m->data[i * m->cols + j] - n->data[i * n->cols + j];
        }
    }
    return out;
}

Matrix* scalarmul(Matrix* m, int x) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[i * m->cols + j] = m->data[i * m->cols + j] * x;
        }
    }
    return out;
}

Matrix* transpose(Matrix* m) {
    Matrix* out = InitMatrix(m->cols, m->rows);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[j * m->rows + i] = m->data[i * m->cols + j];
        }
    }
    return out;
}

Matrix* zeros(int rows, int cols) {
    Matrix* out = InitMatrix(rows, cols);
    int size = rows * cols;
    memset(out->data, 0, size * sizeof(double)); 
    return out;
}

Matrix* ones(int rows, int cols) {
    Matrix* out = InitMatrix(rows, cols);
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            out->data[i * out->cols + j] = 1.0;
        }
    }
    return out;
}

Matrix* identity(int side) {
    Matrix* out = zeros(side, side);
    for (int i = 0, j = 0; i < out->rows && j < out->cols; i++, j++) {
        out->data[i * out->cols + j] = 1.0; 
    }
    return out;
}

Matrix* matmul(Matrix* m, Matrix* n) {
    assert(m->cols == n->rows);
    Matrix* out = InitMatrix(m->rows, n->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < n->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < m->cols; k++) {
                sum += m->data[i * m->cols + k] * n->data[k * n->cols + j];
            }
            out->data[i * out->cols + j] = sum;
        }
    }
    return out;
}

Matrix* slice(Matrix* m, int from_rows, int to_rows, int from_cols, int to_cols) {
    assert(from_rows >= 0 && from_cols >= 0 && to_rows <= m->rows && to_cols <= m->cols);
    int new_rows = to_rows - from_rows;
    int new_cols = to_cols - from_cols;

    Matrix* out = InitMatrix(new_rows, new_cols);
    for (int i = from_rows, out_i = 0; (i < to_rows && out_i < out->rows); i++, out_i++) {
        for (int j = from_cols, out_j = 0; (j < to_cols && out_j < out->cols); j++, out_j++) {
            out->data[out_i * out->cols + out_j] = m->data[i * m->cols + j];
        }
    }
    return out;
}

double determinant(Matrix* m) {
    assert(m->rows == m->cols);
    double out = 0;
    if (m->rows == 1) {
        out = m->data[0];
    } else if (m->rows == 2) {
        out = (m->data[0] * m->data[3]) - (m->data[1] * m->data[2]);
    } else {
       for (int i = 0; i < m->rows; i++) {
           for (int j = 0; j < m->cols; j++) {

           }
       }         
    }
    return out;
}
