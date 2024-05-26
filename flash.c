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

void FreeMatrixTuple(MatrixTuple mt) {
    FreeMatrix(mt.first);
    FreeMatrix(mt.second);
}

void FreeSVDStruct(SVDStruct svd) {
    FreeMatrix(svd.U);
    FreeVector(svd.S);
    FreeMatrix(svd.V);
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

void VectorSet(Vector* v, size_t index, double value) {
    if (index >= v->size) {
        return;
    }
    v->data[index] = value;
}

double VectorGet(Vector* v, size_t index) {
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
    printf(" ]), size=%zu)\n\n", v->size);
}

char* PrintVectorToString(Vector* v) {
    int len = snprintf(NULL, 0, "Vector(data=[");
    for (int i = 0; i < v->size; i++) {
        len += snprintf(NULL, 0, " %f", v->data[i]);
    }
    len += snprintf(NULL, 0, "]), size=%zu)\n\n", v->size);
    
    char* str = malloc(len + 1);
    sprintf(str, "Vector(data=[");
    for (int i = 0; i < v->size; i++) {
        sprintf(str + strlen(str), " %f", v->data[i]);
    }
    sprintf(str + strlen(str), " ]), size=(%zu,))\n\n", v->size);
    return str;
}

Vector* VectorAdd(Vector* v, Vector* w) {
    assert(v->size == w->size);
    Vector* out = InitVector(v->size);

    for (int i = 0; i < out->size; i++) {
        out->data[i] = (v->data[i] + w->data[i]);
    }
    return out;
}

Vector* VectorSub(Vector* v, Vector* w) {
    assert(v->size == w->size);
    Vector* out = InitVector(v->size);

    for (int i = 0; i < out->size; i++) {
        out->data[i] = (v->data[i] - w->data[i]);
    }
    return out;
}

Vector* VectorScale(Vector* v, int x) {
    Vector* out = InitVector(v->size);

    for (int i = 0; i < out->size; i++) {
        out->data[i] = (v->data[i] * x);
    }
    return out;
}

double VectorNorm(Vector* v) {
    double out = 0.0;
    for (int i = 0; i < v->size; i++) {
        out += (v->data[i] * v->data[i]);
    }
    return sqrt(out);
}

double VectorDotProduct(Vector* v, Vector* w) {
    assert(v->size == w->size);
    double out = 0.0;

    for (int i = 0; i < v->size; i++) {
        out += (v->data[i] * w->data[i]);
    }

    return out;
}

double VectorAngle(Vector* v, Vector* w) {
    double norm_v = VectorNorm(v);
    double norm_w = VectorNorm(w);
    double dot_p = VectorDotProduct(v, w);

    double out = dot_p / (norm_v * norm_w);
    return acos(out);
}

double VectorCrossProduct(Vector* v, Vector* w) {
    double norm_v = VectorNorm(v);
    double norm_w = VectorNorm(w);
    double angle = radian_to_degrees(VectorAngle(v, w));

    return (norm_v * norm_w) * sin(angle);
}

bool VectorEqual(Vector* v, Vector* w) {
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

Vector* VectorNormalize(Vector* v) {
    float Norm_v = VectorNorm(v);
    Vector* out = InitVector(v->size);

    for (int i = 0; i < v->size; i++) {
        out->data[i] = (v->data[i] / Norm_v);
    }
    return out;
}

Vector* ZerosVector(size_t size) {
    Vector* out = InitVector(size);
    for (int i = 0; i < size; i++) {
        out->data[i] = 0.0;
    }
    return out;
}

Vector* OnesVector(size_t size) {
    Vector* out = InitVector(size);
    for (int i = 0; i < size; i++) {
        out->data[i] = 1.0;
    }
    return out;
}

Vector* RandomVector(size_t size, int seed) {
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

Vector* VectorCopy(Vector* v) {
    Vector* out = InitVector(v->size);
    for (int i = 0; i < v->size; i++) {
        out->data[i] = v->data[i];
    }
    return out;
}

Vector* VectorMultiply(Vector* v, Vector* w){
    assert(v->size == w->size);
    Vector* out = InitVector(v->size);
    for (int i = 0; i < v->size; i++) {
        out->data[i] = v->data[i] * w->data[i];
    }
    return out;
}

double VectorProjection(Vector* v, Vector* w) {
    double vdotw = VectorDotProduct(v, w);
    double mag_w = VectorNorm(w);
    return vdotw / mag_w;
}

Vector* VectorTransform(Vector* v, Matrix* m) {
    assert(m->cols == v->size);
    Vector* out = ZerosVector(m->rows);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[i] += (m->data[i * m->cols + j] * v->data[j]);
        }
    }
    return out;
}

Vector* VectorOrthog(Vector* v) {
    // TODO: This is not correct
    size_t size = v->size;
    Vector* out = InitVector(size);

    size_t first_nonzero = 0;
    while (first_nonzero < size && VectorGet(v, first_nonzero) == 0.0) {
        first_nonzero++;
    }
    if (first_nonzero == size) {
        VectorSet(out, 0, 1.0);
        return out;
    }

    VectorSet(out, first_nonzero, 1.0);
    double neg_first_nonzero = -VectorGet(v, first_nonzero);
    for (size_t i = 0; i < size; i++) {
        if (i != first_nonzero) {
            VectorSet(out, i, neg_first_nonzero);
        }
    }
    return out;
}

double VectorSum(Vector* v) {
    double sum = 0.0;
    for (int i = 0; i < v->size; i++) {
        sum += v->data[i];
    }
    return sum;
}

Vector* VectorExp(Vector* v) {
    Vector* out = VectorCopy(v);
    for (int i = 0; i < v->size; i++) {
        out->data[i] = exp(v->data[i]);
    }
    return out;
}

bool VectorAllClose(Vector* v, Vector* w) {
    if (v->size != w->size) {
        return false;
    }
    
    double atol = 1e-08;
    double rtol = 1e-05;

    for (int i = 0; i < v->size; i++) {
        if (fabs(v->data[i] - w->data[i]) > (atol + rtol * fabs(w->data[i]))) {
            return false;
        }
    }
    return true;
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

    printf("Matrix(data=(\n[");
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
    printf("), size=(%d, %d))\n\n", m->rows, m->cols);
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

Matrix* MatrixAdd(Matrix* m, Matrix* n) {
    assert(m->rows * m->cols == n->rows * n->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[i * m->cols + j] = m->data[i * m->cols + j] + n->data[i * n->cols + j];
        }
    }
    return out;
}

Matrix* MatrixSub(Matrix* m, Matrix* n) {
    assert(m->rows * m->cols == n->rows * n->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[i * m->cols + j] = m->data[i * m->cols + j] - n->data[i * n->cols + j];
        }
    }
    return out;
}

Matrix* MatrixScale(Matrix* m, int x) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[i * m->cols + j] = m->data[i * m->cols + j] * x;
        }
    }
    return out;
}

Matrix* MatrixTranspose(Matrix* m) {
    Matrix* out = InitMatrix(m->cols, m->rows);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[j * m->rows + i] = m->data[i * m->cols + j];
        }
    }
    return out;
}

Matrix* ZerosMatrix(int rows, int cols) {
    Matrix* out = InitMatrix(rows, cols);
    int size = rows * cols;
    memset(out->data, 0, size * sizeof(double)); 
    return out;
}

Matrix* OnesMatrix(int rows, int cols) {
    Matrix* out = InitMatrix(rows, cols);
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            out->data[i * out->cols + j] = 1.0;
        }
    }
    return out;
}

Matrix* IdentityMatrix(int side) {
    Matrix* out = ZerosMatrix(side, side);
    for (int i = 0, j = 0; i < out->rows && j < out->cols; i++, j++) {
        out->data[i * out->cols + j] = 1.0; 
    }
    return out;
}

Matrix* MatrixMul(Matrix* m, Matrix* n) {
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

Matrix* MatrixSlice(Matrix* m, int from_rows, int to_rows, int from_cols, int to_cols) {
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

MatrixTuple LUDecomposition(Matrix* A) {
    int n = A->rows;
    assert(A->rows == A->cols);

    Matrix* L = InitMatrix(n, n);
    Matrix* U = InitMatrix(n, n);

    for (int i = 0; i < n; i++) {
        // upper triangular
        for (int k = 0; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L->data[i * n + j] * U->data[j * n + k];
            }
            U->data[i * n + k] = A->data[i * n + k] - sum;
        }

        // lower triangular
        for (int k = i + 1; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L->data[k * n + j] * U->data[j * n + i];
            }
            L->data[k * n + i] = (A->data[k * n + i] - sum) / U->data[i * n + i];
        }
    
        // diagonal elements of L 
        L->data[i * n + i] = 1.0;
    }
    return (MatrixTuple){L, U}; 
}

double MatrixDeterminant(Matrix* m) {
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

            out += (i % 2 == 0 ? 1 : -1) * m->data[i] * MatrixDeterminant(submatrix);
            //printf("out = %f | m->data[i] = %f\n", out, m->data[i]);
            FreeMatrix(submatrix);
        }
    }
    return out;
}

double MatrixTrace(Matrix* m) {
    assert(m->rows == m->cols);

    double out = 0.0;
    int i = 0;
    while (i < (m->rows*m->cols)) {
        out += m->data[i];
        i += m->rows+1;
    }
    return out;
}

double FrobeniusNorm(Matrix* m) {
    double out = 0.0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out += (m->data[i * m->cols + j] * m->data[i * m->cols + j]);
        }
    }
    return sqrt(out);
}

double L1Norm(Matrix* m) {
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

double InfiniyNorm(Matrix* m) {
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

double Norm(Matrix* m, char* type) {
    if ((strcmp(type, "frobenius") == 0) || (strcmp(type, "euclidean")) == 0) {
        return FrobeniusNorm(m);
    } else if (strcmp(type, "l1") == 0) {
        return L1Norm(m);
    } else if (strcmp(type, "infinity") == 0){
        return InfiniyNorm(m);
    } else {
        printf("Invalid type: enter 'frobenius', 'euclidean', 'l1', 'infinity'.");
        printf("frobenius Norm: ");
        return FrobeniusNorm(m);
    }
}

Matrix* MatrixConcat(Matrix* m, Matrix* n, int axis) {
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
        return ZerosMatrix(m->rows, m->cols);
    }
}

Matrix* MatrixCopy(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    SetElements(out, m->data);
    return out;
}

Matrix* MatrixNormalize(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    double Norm = FrobeniusNorm(m);

    if (Norm > 0.0) {
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                out->data[i * m->cols + j] = (m->data[i * m->cols + j] / Norm);
            }
        }
    }
    return out;
}

void swap_rows(Matrix* m, int row1, int row2) {
    if (row1 == row2) 
        return;

    for (int i = 0; i < m->cols; i++) {
        double temp = m->data[row1 * m->cols + i];
        m->data[row1 * m->cols + i] = m->data[row2 * m->cols + i];
        m->data[row2 * m->cols + i] = temp;
    }
   }

void mult_row(Matrix* m, int row, double x) {
    for (int j = 0; j < m->cols; j++) {
        m->data[row * m->cols + j] *= x;
    }
}

void add_row(Matrix* m, int row1, int row2, double scalar) {
    for (int j = 0; j < m->cols; j++) {
        m->data[row1 * m->cols + j] += scalar * m->data[row2 * m->cols + j];
    }
}

int find_pivot(Matrix* m, int col, int row) {
    for (int i = row; i < m->rows; i++) {
        if (fabs(m->data[i * m->cols + col]) > 1e-10) {
            return i;
        }
    }
    return -1;
}

Matrix* MatrixRowEchelon(Matrix* m) {
    Matrix* out = MatrixCopy(m);
    int lead = 0;
    int rows = out->rows;
    int cols = out->cols;

    while (lead < rows && lead < cols) {
        int pivot = find_pivot(out, lead, lead);
        if (pivot == -1) {
            lead++;
            continue;
        }

        swap_rows(out, lead, pivot);
        mult_row(out, lead, 1.0 / out->data[lead * cols + lead]);

        for (int i = 0; i < rows; i++) {
            if (i != lead) {
                add_row(out, i, lead, -out->data[i * cols + lead]);
            }
        }

        lead++;
    }
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            if (out->data[i * out->cols + j] == -0.0) {
                out->data[i * out->cols + j] = 0.0;
            }
        }
    }
    return out;
}


Matrix* MatrixInverse(Matrix* m) {
    assert(m->rows == m->cols);
    assert(MatrixDeterminant(m) != 0.0);

    int n = m->rows;
    Matrix* eye = IdentityMatrix(n);
    Matrix* aug = MatrixConcat(m, eye, 0);

    Matrix* row_ech = MatrixRowEchelon(aug);

    Matrix* inv = InitMatrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv->data[i * n + j] = row_ech->data[i * (2*n) + n + j];
        } 
    }
    FreeMatrix(row_ech);
    FreeMatrix(aug);
    FreeMatrix(eye);
    return inv;
}

void matrix_col_copy(Matrix* m, int col, Matrix* dst, int dst_col) {
    for (int i = 0; i < m->rows; i++) {        
        dst->data[i * dst->cols + dst_col] = m->data[i * m->cols + col];     
    }
}

void matrix_col_subtract(Matrix* m, int col, Matrix* dst, int dst_col, double scalar) {
    for (int i = 0; i < m->rows; i++) {
        m->data[i * dst->cols + col] -= scalar * dst->data[i * m->cols + dst_col];
    }
}

void matrix_col_divide(Matrix* m, int col, double scalar) {
    for (int i = 0; i < m->rows; i++) {
        m->data[i * m->cols + col] /= scalar;
    }
}

double vector_length(Matrix* m, int col) {
    double sum = 0.0;
    for (int i = 0; i < m->rows; i++) {
        sum += m->data[i * m->cols + col] * m->data[i * m->cols + col];
    }
    return sqrt(sum);
}

double vector_dot(Matrix* m, int col1, Matrix* n, int col2) {
    double dot = 0;
    for (int i = 0; i < m->rows; i++) {
        dot += m->data[i * m->cols + col1] * n->data[i * n->cols + col2];
    }
    return dot;
}

MatrixTuple QRDecomposition(Matrix* m) {
    assert(m->rows == m->cols);
    assert(MatrixDeterminant(m) != 0.0);

    Matrix* A = MatrixCopy(m);
    Matrix* Q = ZerosMatrix(m->rows, m->cols);
    Matrix* R = ZerosMatrix(m->rows, m->cols);
  
    
    for (int i = 0; i < A->cols; i++) {
        matrix_col_copy(A, i, Q, i);
        for (int j = 0; j < i; j++) {
            double r = vector_dot(Q, i, Q, j); 
            R->data[j * A->cols + i] = r;
            matrix_col_subtract(Q, i, Q, j, r);
        }
        double norm = vector_length(Q, i);
        R->data[i * A->rows + i] = norm;
        matrix_col_divide(Q, i, norm);
    }
    
    FreeMatrix(A);
    return (MatrixTuple){Q, R};
}

Matrix* QRAlgorithm(Matrix* m) {
    MatrixTuple QR; Matrix* Q;
    int MAX_ITER = 500;
    Q = MatrixCopy(m);

    for (int i = 0; i < MAX_ITER; i++) {
        QR = QRDecomposition(Q);
        Q = MatrixMul(QR.second, QR.first);

        if (MatrixMax(MatrixAbs(MatrixTril(Q, -1))) < 1e-10) {
            break;
        }
    }
    return Q;
}

Vector* MatrixEig(Matrix* m) {
    Vector* out = InitVector(m->rows);
    Matrix* Q = QRAlgorithm(m);

    for (int i = 0; i < out->size; i++) {
        out->data[i] = Q->data[i * out->size + i];
    }
    return out;
}

int non_zero_rows(Matrix* m) {
    int count = 0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (m->data[i * m->cols + j] == 0) {
                continue;
            } else if (m->data[i * m->cols + j] != 0 && (i * m->cols + j + 1) % m->cols == 0) {
                count++;
            }
        }
    }
    return count;
}


int MatrixRank(Matrix* m) {
    Matrix* rem;
    int non_zero_row_count = 0;

    if (m->rows == m->cols && MatrixDeterminant(m) != 0) {
        return m->rows;
    } else if (m->rows == m->cols && MatrixDeterminant(m) == 0) {
        rem = MatrixRowEchelon(m);
        non_zero_row_count = non_zero_rows(rem);
        FreeMatrix(rem);
        }
    else {
        rem = MatrixRowEchelon(m);
        non_zero_row_count = non_zero_rows(rem);
        FreeMatrix(rem);
    }
    return non_zero_row_count;
}

SVDStruct SVD(Matrix* m) {
    int x = m->rows, y = m->cols;
    Matrix* U = IdentityMatrix(x);
    Matrix* V = IdentityMatrix(y);
    Matrix* S = MatrixCopy(m);
    int MAX_ITERS = 100;
    MatrixTuple US, SV;

    for (int i = 0; i < MAX_ITERS; i++) {
        US = QRDecomposition(S);
        S = US.second;
        U = MatrixMul(U, US.first);

        SV = QRDecomposition(MatrixTranspose(S));
        S = MatrixTranspose(SV.second);
        V = MatrixMul(V, SV.first);

        if (VectorAllClose(MatrixDiagonal(S, 0), ZerosVector(S->rows))) {
            break;
        }
    }
    Vector* Sigma = MatrixDiagonal(S, 0);
    return (SVDStruct){U, Sigma, V};
}

Vector* MatrixDiagonal(Matrix* m, int k) {
    assert(m->rows == m->cols);
    Vector* out = InitVector(m->rows - abs(k));
    int start = (k >= 0) ? 0 : -k;
    for (int i = start, j = 0; i < m->rows - abs(k) && j < out->size; i++, j++) {
        out->data[j] = m->data[i * m->rows + i + k];
    }
    return out;
}

Matrix* MatrixTril(Matrix* m, int diag) {
    assert(m->rows == m->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            if (j <= i - diag) {
                out->data[i * out->cols + j] = m->data[i * m->cols + j];
            } else {
                out->data[i * out->cols + j] = 0.0;
            }
        }
    }
    return out;
}

Matrix* MatrixTriu(Matrix* m, int diag) {
    assert(m->rows == m->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            if (j >= i + diag) {
                out->data[i * out->cols + j] = m->data[i * m->cols + j];
            } else {
                out->data[i * out->cols + j] = 0.0;
            }
        }
    }
    return out;
}
double MatrixMax(Matrix* m) {
    double max_val = -INFINITY;
    
    for (int i = 0; i < m->rows*m->cols; i++) {
        if (m->data[i] > max_val) {
            max_val = m->data[i];
        }
    }
    return max_val;
}

double MatrixMin(Matrix* m) {
    double min_val = INFINITY;
    
    for (int i = 0; i < m->rows*m->cols; i++) {
        if (m->data[i] < min_val) {
            min_val = m->data[i];
        }
    }
    return min_val;
}

double MatrixMean(Matrix* m) {
    double out = 0.0;
    int num_el = m->rows * m->cols;

    for (int i = 0; i < num_el; i++) {
        out += m->data[i];
    }
    return out / num_el;
}

double MatrixStd(Matrix* m) {
    double mean = MatrixMean(m);
    int num_el = m->rows * m->cols;
    double out = 0.0;

    for (int i = 0; i < num_el; i++) {
        out += (m->data[i] - mean)*(m->data[i] - mean);
    }
    return sqrt(out / num_el);
}

Matrix* MatrixMaxVals(Matrix* m, int dim) {
    double max_val = -INFINITY;

    if (dim == 0) {
        Matrix* out_m = InitMatrix(m->rows, 1);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                if (m->data[i * m->rows + j] > max_val) {
                    max_val = m->data[i * m->rows + j];
                }
            }
            out_m->data[i] = max_val;
            max_val = -INFINITY;
        }
        return out_m;
    } else if (dim == 1) {
        Matrix* out_m = InitMatrix(1, m->cols);
        for (int j = 0; j < m->cols; j++) {
            for (int i = 0; i < m->rows; i++) {
                if (m->data[i * m->cols + j] > max_val) {
                    max_val = m->data[i * m->cols + j];
                }
            }
            out_m->data[j] = max_val;
            max_val = -INFINITY;
        }
        return out_m;
    } else {
        return NULL;
    }
}

Matrix* MatrixMinVals(Matrix* m, int dim) {
    double min_val = INFINITY;

    if (dim == 0) {
        Matrix* out_m = InitMatrix(m->rows, 1);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                if (m->data[i * m->rows + j] < min_val) {
                    min_val = m->data[i * m->rows + j];
                }
            }
            out_m->data[i] = min_val;
            min_val = INFINITY;
        }
        return out_m;
    } else if (dim == 1) {
        Matrix* out_m = InitMatrix(1, m->cols);
        for (int j = 0; j < m->cols; j++) {
            for (int i = 0; i < m->rows; i++) {
                if (m->data[i * m->cols + j] < min_val) {
                    min_val = m->data[i * m->cols + j];
                }
            }
            out_m->data[j] = min_val;
            min_val = INFINITY;
        }
        return out_m;
    } else {
        return NULL;
    }
}

Matrix* MatrixMeanVals(Matrix* m, int dim) {
    int row_elem = m->cols;
    int col_elem = m->rows;

    if (dim == 0) {
        double row_sum = 0.0;
        Matrix* out_m = InitMatrix(m->rows, 1);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                row_sum += m->data[i * m->cols + j];
            }
            out_m->data[i] = row_sum / row_elem;
            row_sum = 0.0;
        }
        return out_m;
    } else if (dim == 1) {
        double col_sum = 0.0;
        Matrix* out_m = InitMatrix(1, m->cols);
        for (int j = 0; j < m->cols; j++) {
            for (int i = 0; i < m->rows; i++) {
                col_sum += m->data[i * m->cols + j];
            }
            out_m->data[j] = col_sum / col_elem;
            col_sum = 0.0;
        }
        return out_m;
    } else {
        return NULL;
    }
}

Matrix* MatrixStdVals(Matrix* m, int dim) {
    int row_elem = m->cols;
    int col_elem = m->rows;

    if (dim == 0) {
        double row_sum = 0.0;
        Matrix* out_m = InitMatrix(m->rows, 1);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                row_sum += m->data[i * m->cols + j] * m->data[i * m->cols + j];
            }
            out_m->data[i] = sqrt(row_sum / row_elem);
            row_sum = 0.0;
        }
        return out_m;
    } else if (dim == 1) {
        double col_sum = 0.0;
        Matrix* out_m = InitMatrix(1, m->cols);
        for (int j = 0; j < m->cols; j++) {
            for (int i = 0; i < m->rows; i++) {
                col_sum += m->data[i * m->cols + j] * m->data[i * m->cols + j];
            }
            out_m->data[j] = sqrt(col_sum / col_elem);
            col_sum = 0.0;
        }
        return out_m;
    } else {
        return NULL;
    }
}

bool MatrixAllClose(Matrix* m, Matrix* n) {
    if (m->rows != n->rows || m->cols != n->cols) {
        return false;
    }
    
    double atol = 1e-08;
    double rtol = 1e-05;

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (fabs(m->data[i * m->cols + j] - n->data[i * n->cols + j]) > (atol + rtol * fabs(n->data[i * n->cols + j]))) {
                return false;
            }
        }
    }
    return true;
}

Matrix* MatrixSolve(Matrix* m, Matrix* n) {
    Matrix* m_inv = MatrixInverse(m);
    return MatrixMul(MatrixTranspose(n), m_inv);
}

Matrix* MatrixAbs(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            out->data[i * out->cols + j] = fabs(m->data[i * out->cols + j]);
        }
    }
    return out;
}

Matrix* CholeskyDecomposition(Matrix* m) {
    assert(MatrixAllClose(m, MatrixTranspose(m)));
    Matrix* L = ZerosMatrix(m->rows, m->cols);
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j <= i; j++) {
            double sum_val = 0.0;
            for (int k = 0; k < j; k++) {
                sum_val += L->data[i * L->cols + k] * L->data[j * L->cols + k];
            }
            if (i == j) {
                L->data[i * L->cols + j] = sqrt(m->data[i * m->cols + j] - sum_val);
            } else {
                L->data[i * L->cols + j] = (1.0 / L->data[j * L->cols + j]) * (m->data[i * m->cols + j] - sum_val); 
            }
        }
    }
    return L;
}
