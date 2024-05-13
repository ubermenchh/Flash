#include "flash.h"

/*
*****************************************************
***************** HELPER FUNCTIONS ******************
*****************************************************
*/
double radian_to_degrees(double x) {
    /* Converts radian to degree */
    return (x * 180) / (double)M_PI;
}

/*
*****************************************************
***************** VECTOR FUNCTIONS ******************
*****************************************************
*/

Vector* InitVector(size_t size) {
    Vector* vector = (Vector*)malloc(sizeof(Vector));
    vector->size = size;
    vector->data = (double*)calloc(size, sizeof(double));
    return vector;
}

void FreeVector(Vector* v) {
    free(v->data);
    free(v);
}

void VectorSetElements(Vector* v, double* values) {
    for (int i = 0; i < v->size; i++){
        v->data[i] = values[i];
    }
}

void vector_set(Vector* v, size_t index, double value) {
    if (index >= v->size) {
        return;
    }
    v->data[index] = value;
}

double vector_get(Vector* v, size_t index) {
    if (index >= v->size) {
        return 0.0;
    }
    return v->data[index];
}

void PrintVector(Vector* v) {
    printf("Vector(data=([");
    for (int i = 0; i < v->size; i++) {
        printf(" %f", v->data[i]);
    }
    printf(" ]), size=%zu)\n", v->size);
}

Vector* vector_add(Vector* v, Vector* w) {
    assert(v->size == w->size);
    Vector* out = InitVector(v->size);

    for (int i = 0; i < out->size; i++) {
        out->data[i] = (v->data[i] + w->data[i]);
    }
    return out;
}

Vector* vector_sub(Vector* v, Vector* w) {
    assert(v->size == w->size);
    Vector* out = InitVector(v->size);

    for (int i = 0; i < out->size; i++) {
        out->data[i] = (v->data[i] - w->data[i]);
    }
    return out;
}

Vector* vector_scale(Vector* v, int x) {
    Vector* out = InitVector(v->size);

    for (int i = 0; i < out->size; i++) {
        out->data[i] = (v->data[i] * x);
    }
    return out;
}

double vector_norm(Vector* v) {
    double out = 0.0;
    for (int i = 0; i < v->size; i++) {
        out += (v->data[i] * v->data[i]);
    }
    return sqrt(out);
}

double vector_dotproduct(Vector* v, Vector* w) {
    assert(v->size == w->size);
    double out = 0.0;

    for (int i = 0; i < v->size; i++) {
        out += (v->data[i] * w->data[i]);
    }

    return out;
}

double vector_angle(Vector* v, Vector* w) {
    double norm_v = vector_norm(v);
    double norm_w = vector_norm(w);
    double dot_p = vector_dotproduct(v, w);

    double out = dot_p / (norm_v * norm_w);
    return acos(out);
}

double vector_crossproduct(Vector* v, Vector* w) {
    double norm_v = vector_norm(v);
    double norm_w = vector_norm(w);
    double angle = radian_to_degrees(vector_angle(v, w));

    return (norm_v * norm_w) * sin(angle);
}

bool vector_equal(Vector* v, Vector* w) {
    if (v->size != w->size) {
        return false;
    }
    for (int i = 0; i < v->size; i++) {
        if (v->data[i] != w->data[i]) {
            return false;
        }
    }
    return true;
}

Vector* vector_normalize(Vector* v) {
    float norm_v = vector_norm(v);
    Vector* out = InitVector(v->size);

    for (int i = 0; i < v->size; i++) {
        out->data[i] = (v->data[i] / norm_v);
    }
    return out;
}

Vector* zeros_vector(size_t size) {
    Vector* out = InitVector(size);
    for (int i = 0; i < size; i++) {
        out->data[i] = 0.0;
    }
    return out;
}

Vector* ones_vector(size_t size) {
    Vector* out = InitVector(size);
    for (int i = 0; i < size; i++) {
        out->data[i] = 1.0;
    }
    return out;
}

Vector* random_vector(size_t size, int seed) {
    if (seed != 0) {
        srand(seed);
    } else {
        srand(time(NULL));
    }
    Vector* out = InitVector(size);
    for (int i = 0; i < size; i++) {
        out->data[i] = (double)rand() / (double)RAND_MAX;
    }
    return out;
}

Vector* vector_copy(Vector* v) {
    Vector* out = InitVector(v->size);
    for (int i = 0; i < v->size; i++) {
        out->data[i] = v->data[i];
    }
    return out;
}

Vector* vector_multiply(Vector* v, Vector* w){
    assert(v->size == w->size);
    Vector* out = InitVector(v->size);
    for (int i = 0; i < v->size; i++) {
        out->data[i] = v->data[i] * w->data[i];
    }
    return out;
}

double vector_projection(Vector* v, Vector* w) {
    double vdotw = vector_dotproduct(v, w);
    double mag_w = vector_norm(w);
    return vdotw / mag_w;
}

/*
******************************************
*************** MATRICES *****************
******************************************
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

void LUDecomp(Matrix* A, Matrix** L, Matrix** U) {
    int n = A->rows;
    assert(A->rows == A->cols);

    *L = InitMatrix(n, n);
    *U = InitMatrix(n, n);

    for (int i = 0; i < n; i++) {
        // upper triangular
        for (int k = 0; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += (*L)->data[i * n + j] * (*U)->data[j * n + k];
            }
            (*U)->data[i * n + k] = A->data[i * n + k] - sum;
        }

        // lower triangular
        for (int k = i + 1; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += (*L)->data[k * n + j] * (*U)->data[j * n + i];
            }
            (*L)->data[k * n + i] = (A->data[k * n + i] - sum) / (*U)->data[i * n + i];
        }
    
        // diagonal elements of L 
        (*L)->data[i * n + i] = 1.0;
    }
}

double determinant(Matrix* m) {
    assert(m->rows == m->cols);
    int n = m->rows;
    double out = 0.0;

    if (n == 1) {
        out = m->data[0];
    } else if (n == 2) {
        out = (m->data[0] * m->data[3]) - (m->data[1] * m->data[2]);
        //printf("%f\n", out);
    } else {
        for (int i = 0; i < n; i++) {
            Matrix* submatrix = InitMatrix(n-1, n-1);
            int sub_row = 0, sub_col = 0;

            for (int j = 1; j < n; j++) {
                if (j != 0) {
                    for (int k = 0; k < n; k++) {
                        if (k != i) {
                            submatrix->data[sub_row * (n-1) + sub_col] = m->data[j * n + k];
                            sub_col++;
                            if (sub_col == n-1) {
                                sub_col = 0;
                                sub_row++;
                            }
                        }
                    }
                } 
            }
            //printf("Sub-Matrix for row %d\n", i);
            //PrintMatrix(submatrix);

            out += (i % 2 == 0 ? 1 : -1) * m->data[i] * determinant(submatrix);
            //printf("out = %f | m->data[i] = %f\n", out, m->data[i]);
            FreeMatrix(submatrix);
        }
    }
    return out;
}

double trace(Matrix* m) {
    assert(m->rows == m->cols);

    double out = 0.0;
    int i = 0;
    while (i < (m->rows*m->cols)) {
        out += m->data[i];
        i += m->rows+1;
    }
    return out;
}

double frobenius_norm(Matrix* m) {
    double out = 0.0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out += (m->data[i * m->cols + j] * m->data[i * m->cols + j]);
        }
    }
    return sqrt(out);
}

double l1_norm(Matrix* m) {
    double out = 0.0;

    for (int i = 0; i < m->rows; i++) {
        double col_sum = 0.0;
        for (int j = 0; j < m->cols; j++) {
            col_sum += fabs(m->data[j * m->rows + i]);
        }
        if (col_sum > out) {
            out = col_sum;
        }
    }
    return out;
}

double infinity_norm(Matrix* m) {
    double out = 0.0;

    for (int i = 0; i < m->rows; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < m->cols; j++) {
            row_sum += fabs(m->data[i * m->rows + j]);
        }
        if (row_sum > out) {
            out = row_sum;
        }
    }
    return out; 
}

double norm(Matrix* m, char* type) {
    if ((strcmp(type, "frobenius") == 0) || (strcmp(type, "euclidean")) == 0) {
        return frobenius_norm(m);
    } else if (strcmp(type, "l1") == 0) {
        return l1_norm(m);
    } else if (strcmp(type, "infinity") == 0){
        return infinity_norm(m);
    } else {
        printf("Invalid type: enter 'frobenius', 'euclidean', 'l1', 'infinity'.");
        printf("frobenius Norm: ");
        return frobenius_norm(m);
    }
}

Matrix* concat(Matrix* m, Matrix* n, int axis) {
    if (axis == 0) {
        assert(m->rows == n->rows);
        int new_cols = m->cols + n->cols;
        
        Matrix* out = InitMatrix(m->rows, new_cols);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                out->data[i * new_cols + j] = m->data[i * m->cols + j];    
            }
        }
        for (int i = 0; i < n->rows; i++) {
            for (int j = 0; j < n->cols; j++) {
                out->data[i * new_cols + (j+m->rows)] = n->data[i * n->cols + j];
            }
        }
        return out;
    } else if (axis == 1) {
        assert(m->cols == n->cols);
        int new_rows = m->rows + n->rows;
        
        Matrix* out = InitMatrix(new_rows, m->cols);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                out->data[i * m->rows + j] = m->data[i * m->rows + j];
            }
        }
        for (int i = 0; i < n->rows; i++) {
            for (int j = 0; j < n->cols; j++) {
                out->data[(i) * m->rows + (j+(m->rows*m->cols))] = n->data[i * n->rows + j];
            }
        }
        return out;

    } else {
        printf("Invaid axis: Use `0` for row-wise concatenation and `1` for column-wise.");
        return zeros(m->rows, m->cols);
    }
}

Matrix* copy(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    SetElements(out, m->data);
    return out;
}
