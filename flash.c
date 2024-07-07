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
            out->data[i] += (MAT_AT(m, i, j) * v->data[j]);
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

    for (int i = 0; i < v->size; i++) {
        if (fabs(v->data[i] - w->data[i]) > (ATOL + RTOL * fabs(w->data[i]))) {
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
    
#pragma omp parallel for
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double val = fabs(MAT_AT(m, i, j));
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
            printf("%*.*f ", max_digits, 5, MAT_AT(m, i, j));
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
        srand(seed);
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
    int out_rows = (m->rows > n->rows) ? m->rows : n->rows;
    int out_cols = (m->cols > n->cols) ? m->cols : n->cols;

    assert(m->rows == n->rows || m->rows == 1 || n->rows == 1);
    assert(m->cols == n->cols || m->cols == 1 || n->cols == 1);

    Matrix* out = InitMatrix(out_rows, out_cols);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double m_val = MAT_AT(m, (m->rows == 1) ? 0 : i, (m->cols == 1) ? 0 : j);
            double n_val = MAT_AT(n, (n->rows == 1) ? 0 : i, (n->cols == 1) ? 0 : j);
           MAT_AT(out, i, j) = m_val + n_val;
        }
    }
    return out;
}

Matrix* MatrixSub(Matrix* m, Matrix* n) {
    int out_rows = (m->rows > n->rows) ? m->rows : n->rows;
    int out_cols = (m->cols > n->cols) ? m->cols : n->cols;

    assert(m->rows == n->rows || m->rows == 1 || n->rows == 1);
    assert(m->cols == n->cols || m->cols == 1 || n->cols == 1);

    Matrix* out = InitMatrix(out_rows, out_cols);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double m_val = MAT_AT(m, (m->rows == 1) ? 0 : i, (m->cols == 1) ? 0 : j);
            double n_val = MAT_AT(n, (n->rows == 1) ? 0 : i, (n->cols == 1) ? 0 : j);
           MAT_AT(out, i, j) = m_val - n_val;
        }
    }
    return out;}

Matrix* MatrixScale(Matrix* m, int x) {
    Matrix* out = InitMatrix(m->rows, m->cols);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = MAT_AT(m, i, j) * x;
        }
    }
    return out;
}

Matrix* MatrixTranspose(Matrix* m) {
    Matrix* out = InitMatrix(m->cols, m->rows);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, j, i) = MAT_AT(m, i, j);
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
#pragma omp parallel for collapse(2)
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) = 1.0;
        }
    }
    return out;
}

Matrix* IdentityMatrix(int side) {
    Matrix* out = ZerosMatrix(side, side);
    for (int i = 0, j = 0; i < out->rows && j < out->cols; i++, j++) {
        MAT_AT(out, i, j) = 1.0; 
    }
    return out;
}

Matrix* MatrixMask(int rows, int cols, double prob) {
    Matrix* out = OnesMatrix(rows, cols);
    if (out == NULL) return NULL;

    srand(time(NULL));
    int total_elems = rows*cols;
    int num_zeros = (int)(total_elems * prob);
    
    for (int i = 0; i < num_zeros; i++) {
        int index;
        do {
            index = rand() % total_elems;
        } while (out->data[index] == 0.0);
        out->data[index] = 0.0;
    }
    return out;
}

Matrix* MatrixMul(Matrix* m, Matrix* n) {
    assert(m->cols == n->rows);
    Matrix* out = InitMatrix(m->rows, n->cols);
    
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < n->cols; j++) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (int k = 0; k < m->cols; k++) {
                sum += MAT_AT(m, i, k) * MAT_AT(n, k, j);
            }
            MAT_AT(out, i, j) = sum;
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
            MAT_AT(out, out_i, out_j) = MAT_AT(m, i, j);
        }
    }
    return out;
}

MatrixTuple LUDecomposition(Matrix* A) {
    int n = A->rows;
    assert(A->rows == A->cols);

    Matrix* L = InitMatrix(n, n);
    Matrix* U = InitMatrix(n, n);
#pragma omp parallel 
    for (int i = 0; i < n; i++) {
        // upper triangular
#pragma omp for schedule(dynamic)
        for (int k = 0; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += MAT_AT(L, i, j) * MAT_AT(U, j, k);
            }
            MAT_AT(U, i, k) = MAT_AT(A, i, k) - sum;
        }

        // lower triangular
#pragma omp for schedule(dynamic)
        for (int k = i + 1; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += MAT_AT(L, k, j) * MAT_AT(U, j, i);
            }
            MAT_AT(L, k, i) = (MAT_AT(A, k, i) - sum) / MAT_AT(U, i, i);
        }
    
        // diagonal elements of L
#pragma omp single
        MAT_AT(L, i, i) = 1.0;
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
                            MAT_AT(submatrix, sub_row, sub_col) = MAT_AT(m, j, k);
                            sub_col++;
                            if (sub_col == n-1) {
                                sub_col = 0;
                                sub_row++;
                            }
                        }
                    }
                } 
            }

            out += (i % 2 == 0 ? 1 : -1) * m->data[i] * MatrixDeterminant(submatrix);
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
#pragma omp parallel for
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out += MAT_AT(m, i, j) * MAT_AT(m, i, j);
        }
    }
    return sqrt(out);
}

double L1Norm(Matrix* m) {
    double out = 0.0;

#pragma omp parallel for
    for (int i = 0; i < m->rows; i++) {
        double col_sum = 0.0;
        for (int j = 0; j < m->cols; j++) {
            col_sum += fabs(MAT_AT(m, j, i));
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
            row_sum += fabs(MAT_AT(m, i, j));
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
                MAT_AT(out, i, j) = MAT_AT(m, i, j);    
            }
        }
        for (int i = 0; i < n->rows; i++) {
            for (int j = 0; j < n->cols; j++) {
                MAT_AT(out, i, j+m->rows) = MAT_AT(n, i, j);
            }
        }
        return out;
    } else if (axis == 1) {
        assert(m->cols == n->cols);
        int new_rows = m->rows + n->rows;
        
        Matrix* out = InitMatrix(new_rows, m->cols);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                MAT_AT(out, i, j) = MAT_AT(m, i, j);
            }
        }
        for (int i = 0; i < n->rows; i++) {
            for (int j = 0; j < n->cols; j++) {
                MAT_AT(out, i, (j+(m->rows*m->cols))) = MAT_AT(n, i, j);
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
                MAT_AT(out, i, j) = (MAT_AT(m, i, j) / Norm);
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
        MAT_AT(m, row1, i) = MAT_AT(m, row2, i);
        MAT_AT(m, row2, i) = temp;
    }
   }

void mult_row(Matrix* m, int row, double x) {
    for (int j = 0; j < m->cols; j++) {
        MAT_AT(m, row, j) *= x;
    }
}

void add_row(Matrix* m, int row1, int row2, double scalar) {
    for (int j = 0; j < m->cols; j++) {
        MAT_AT(m, row1, j) += scalar * MAT_AT(m, row2, j);
    }
}

int find_pivot(Matrix* m, int col, int row) {
    for (int i = row; i < m->rows; i++) {
        if (fabs(MAT_AT(m, i, col)) > 1e-10) {
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
        mult_row(out, lead, 1.0 / MAT_AT(out, lead, lead));

        for (int i = 0; i < rows; i++) {
            if (i != lead) {
                add_row(out, i, lead, -MAT_AT(out, i, lead));
            }
        }

        lead++;
    }
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            if (MAT_AT(out, i, j) == -0.0) {
                MAT_AT(out, i, j) = 0.0;
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
            MAT_AT(inv, i, j) = row_ech->data[i * (2*n) + n + j];
        } 
    }
    FreeMatrix(row_ech);
    FreeMatrix(aug);
    FreeMatrix(eye);
    return inv;
}

void matrix_col_copy(Matrix* m, int col, Matrix* dst, int dst_col) {
    for (int i = 0; i < m->rows; i++) {        
        MAT_AT(dst, i, dst_col) = MAT_AT(m, i, col);     
    }
}

void matrix_col_subtract(Matrix* m, int col, Matrix* dst, int dst_col, double scalar) {
    for (int i = 0; i < m->rows; i++) {
        MAT_AT(m, i, col) -= scalar * MAT_AT(dst, i, dst_col);
    }
}

void matrix_col_divide(Matrix* m, int col, double scalar) {
    for (int i = 0; i < m->rows; i++) {
        MAT_AT(m, i, col) /= scalar;
    }
}

double vector_length(Matrix* m, int col) {
    double sum = 0.0;
    for (int i = 0; i < m->rows; i++) {
        sum += MAT_AT(m, i, col) * MAT_AT(m, i, col);
    }
    return sqrt(sum);
}

double vector_dot(Matrix* m, int col1, Matrix* n, int col2) {
    double dot = 0;
    for (int i = 0; i < m->rows; i++) {
        dot += MAT_AT(m, i, col1) * MAT_AT(n, i, col2);
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
            MAT_AT(R, j, i) = r;
            matrix_col_subtract(Q, i, Q, j, r);
        }
        double norm = vector_length(Q, i);
        MAT_AT(R, i, i) = norm;
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
        out->data[i] = MAT_AT(Q, i, i);
    }
    FreeMatrix(Q);
    return out;
}

int non_zero_rows(Matrix* m) {
    int count = 0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (m->data[i * m->cols + j] == 0) {
                continue;
            } else if (MAT_AT(m, i, j) != 0 && (i * m->cols + j + 1) % m->cols == 0) {
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
    FreeMatrix(S);
    FreeMatrixTuple(US);
    FreeMatrixTuple(SV);
    return (SVDStruct){U, Sigma, MatrixTranspose(V)};
}

Vector* MatrixDiagonal(Matrix* m, int k) {
    assert(m->rows == m->cols);
    assert(abs(k) < m->rows);
    Vector* out = InitVector(m->rows - abs(k));

    if (k >= 0) {
        for (int i = 0; i < (m->rows - abs(k)); i++) {
            out->data[i] = MAT_AT(m, i, i + abs(k));
        }
    } else if (k < 0) {
        for (int i = 0; i < (m->rows - abs(k)); i++) {
            out->data[i] = MAT_AT(m, i + abs(k), i);
        }
    }

    return out;
}

Matrix* MatrixTril(Matrix* m, int diag) {
    assert(m->rows == m->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            if (j <= i - diag) {
                MAT_AT(out, i, j) = MAT_AT(m, i, j);
            } else {
                MAT_AT(out, i, j) = 0.0;
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
                MAT_AT(out, i, j) = MAT_AT(m, i, j);
            } else {
                MAT_AT(out, i, j) = 0.0;
            }
        }
    }
    return out;
}

double MatrixSum(Matrix* m) {
    double sum = 0.0;

    for (int i = 0; i < m->rows*m->cols; i++) {
        sum += m->data[i];
    }
    return sum;
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

int MatrixArgMax(Matrix* m) {
    for (int i = 0; i < m->rows*m->cols; i++) {
        if (m->data[i] == MatrixMax(m)) {
            return i;
        }
    }
    return -1;
}

int MatrixArgMin(Matrix* m) {
    for (int i = 0; i < m->rows*m->cols; i++) {
        if (m->data[i] == MatrixMin(m)) {
            return i;
        }
    }
    return -1;
}

Matrix* MatrixSumVals(Matrix* m, int dim) {
    double sum = 0.0;
    if (dim == 0) {
        Matrix* out_m = InitMatrix(m->rows, 1);
        for (int i = 0; i <  m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                sum += MAT_AT(m, i, j);
            }
            out_m->data[i] = sum;
            sum = 0.0;
        }
        return out_m;
    } else if (dim == 1) {
        Matrix* out_m = InitMatrix(1, m->cols);
        for (int j = 0; j < m->cols; j++) {
            for (int i = 0; i < m->rows; i++) {
                sum += MAT_AT(m, i, j);
            }
            out_m->data[j] = sum;
            sum = 0.0;
        }
        return out_m;
    } else {
        return NULL;
    }
}

Matrix* MatrixMaxVals(Matrix* m, int dim) {
    double max_val = -INFINITY;

    if (dim == 0) {
        Matrix* out_m = InitMatrix(m->rows, 1);
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                if (MAT_AT(m, i, j) > max_val) {
                    max_val = MAT_AT(m, i, j);
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
                if (MAT_AT(m, i, j) > max_val) {
                    max_val = MAT_AT(m, i, j);
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
                if (MAT_AT(m, i, j) < min_val) {
                    min_val = MAT_AT(m, i, j); 
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
                if (MAT_AT(m, i, j) < min_val) {
                    min_val = MAT_AT(m, i, j);                }
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
                row_sum += MAT_AT(m, i, j); 
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
                col_sum += MAT_AT(m, i, j);
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
                row_sum += MAT_AT(m, i, j) * MAT_AT(m, i, j);
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
                col_sum += MAT_AT(m, i, j) * MAT_AT(m, i, j);
            }
            out_m->data[j] = sqrt(col_sum / col_elem);
            col_sum = 0.0;
        }
        return out_m;
    } else {
        return NULL;
    }
}

Matrix* MatrixArgMaxVals(Matrix* m, int dim) {
    assert(dim >= 0 && dim < 2);

    if (dim == 0) {
        Matrix* out = InitMatrix(m->rows, 1);
        Matrix* max_m = MatrixMaxVals(m, 0);

        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                if (max_m->data[i] == MAT_AT(m, i, j)) {
                    out->data[i] = j;
                }
            }
        }
        free(max_m);
        return out;
    } else if (dim == 1) {
        Matrix* out = InitMatrix(1, m->cols);
        Matrix* max_m = MatrixMaxVals(m, 1);

        for (int j = 0; j < m->cols; j++) {
            for (int i = 0; i < m->rows; i++) {
                if (max_m->data[j] == MAT_AT(m, i, j)) {
                    out->data[j] = i;
                }
            }
        }
        free(max_m);
        return out;
    } else {
        return NULL;
    }
}

Matrix* MatrixArgMinVals(Matrix* m, int dim) {
    assert(dim >= 0 && dim < 2);

    if (dim == 0) {
        Matrix* out = InitMatrix(m->rows, 1);
        Matrix* min_m = MatrixMinVals(m, 0);

        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->cols; j++) {
                if (min_m->data[i] == MAT_AT(m, i, j)) {
                    out->data[i] = j;
                }
            }
        }
        free(min_m);
        return out;
    } else if (dim == 1) {
        Matrix* out = InitMatrix(1, m->cols);
        Matrix* min_m = MatrixMinVals(m, 1);

        for (int j = 0; j < m->cols; j++) {
            for (int i = 0; i < m->rows; i++) {
                if (min_m->data[j] == MAT_AT(m, i, j)) {
                    out->data[j] = i;
                }
            }
        }
        free(min_m);
        return out;
    } else {
        return NULL;
    }
}

bool MatrixAllClose(Matrix* m, Matrix* n) {
    if (m->rows != n->rows || m->cols != n->cols) {
        return false;
    }

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (fabs(MAT_AT(m, i, j) - MAT_AT(n, i, j)) > (ATOL + RTOL * fabs(MAT_AT(n, i, j)))) {
                return false;
            }
        }
    }
    return true;
}

Matrix* MatrixSolve(Matrix* m, Matrix* n) {
    Matrix* m_inv = MatrixInverse(m);
    Matrix* out = MatrixMul(MatrixTranspose(n), m_inv);
    FreeMatrix(m_inv);
    return out;
}

Matrix* MatrixAbs(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) = fabs(MAT_AT(m, i, j));
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
                sum_val += MAT_AT(L, i, k) * MAT_AT(L, j, k);
            }
            if (i == j) {
                MAT_AT(L, i, j) = sqrt(MAT_AT(m, i, j) - sum_val);
            } else {
                MAT_AT(L, i, j) = (1.0 / MAT_AT(L, j, j) * (MAT_AT(m, i, j) - sum_val)); 
            }
        }
    }
    return L;
}

Matrix* MatrixEigVec(Matrix* m) {
    Matrix* eig = InitMatrix(m->rows, m->cols);
    Vector* eigenvalues = MatrixEig(m);
    Matrix* A; SVDStruct svd;
    Matrix* eigenvector;

    for (int i = 0; i < eigenvalues->size; i++) {
        A = MatrixSub(m, MatrixScalarMul(IdentityMatrix(m->rows), eigenvalues->data[i]));
        svd = SVD(A);
        eigenvector = MatrixSlice(svd.V, 2, m->rows, 0, 3);

        double norm = FrobeniusNorm(eigenvector);
        for (int j = 0; j < m->rows; j++) {
            MAT_AT(eig, j, i) = eigenvector->data[j] / norm;
        }
    }

    FreeVector(eigenvalues);
    FreeMatrix(A);
    FreeMatrix(eigenvector);
    FreeSVDStruct(svd);

    return eig;
}

Matrix* ToMatrix(Vector* v) {
    Matrix* out = InitMatrix(1, v->size);

    for (int i = 0; i < v->size; i++) {
        out->data[i] = v->data[i];
    }
    return out;
}

Matrix* MatrixVectorMul(Matrix* m, Vector* v) {
    assert(m->cols == v->size);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) = MAT_AT(m, i, j) * v->data[j];
        }
    }
    return out;
}

Matrix* MatrixScalarAdd(Matrix* m, double x) {
    Matrix* out = MatrixCopy(m);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) += x;
        }
    }
    return out;
}

Matrix* MatrixScalarSub(Matrix* m, double x) {
    Matrix* out = MatrixCopy(m);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) -= x;
        }
    }
    return out;
}

Matrix* MatrixScalarMul(Matrix* m, double x) {
    Matrix* out = MatrixCopy(m);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) *= x;
        }
    }
    return out;
}

Matrix* MatrixScalarDiv(Matrix* m, double x) {
    Matrix* out = MatrixCopy(m);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) /= x;
        }
    }
    return out;
}

Matrix* MatrixMultiply(Matrix* m, Matrix* n) {
    int out_rows = (m->rows > n->rows) ? m->rows : n->rows;
    int out_cols = (m->cols > n->cols) ? m->cols : n->cols;

    assert(m->rows == n->rows || m->rows == 1 || n->rows == 1);
    assert(m->cols == n->cols || m->cols == 1 || n->cols == 1);

    Matrix* out = InitMatrix(out_rows, out_cols);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double m_val = MAT_AT(m, (m->rows == 1) ? 0 : i, (m->cols == 1) ? 0 : j);
            double n_val = MAT_AT(n, (n->rows == 1) ? 0 : i, (n->cols == 1) ? 0 : j);
           MAT_AT(out, i, j) = m_val * n_val;
        }
    }
    return out;
}

Matrix* MatrixDivide(Matrix* m, Matrix* n) {
    assert(m->rows == n->rows && m->cols == n->cols);
    Matrix* out = MatrixCopy(m);
    
    for (int i = 0; i < out->rows*out->cols; i++) {
        out->data[i] /= n->data[i];
    }
    return out;
}

double MatrixLogDeterminant(Matrix* m) {
    return log(MatrixDeterminant(m));
}

Matrix* MatrixPower(Matrix* m, double exp) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = pow(MAT_AT(m, i, j), exp);
        }
    }
    return out;
}

Matrix* MatrixOnesLike(Matrix* m) {
    return OnesMatrix(m->rows, m->cols);
}

Matrix* MatrixZerosLike(Matrix* m) {
    return ZerosMatrix(m->rows, m->cols);
}

Matrix* MatrixFull(int rows, int cols, double value) {
    Matrix* out = InitMatrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            MAT_AT(out, i, j) = value;
        }
    }
    return out;
}

Matrix* MatrixFullLike(Matrix* m, double value) {
    return MatrixFull(m->rows, m->cols, value);
}

void MatrixFill(Matrix* m, double value) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(m, i, j) = value;
        }
    }
}

Matrix* MatrixReshape(Matrix* m, int rows, int cols) {
    assert(m->rows * m->cols == rows * cols);
    Matrix* out = InitMatrix(rows, cols);

    for (int i = 0; i < rows*cols; i++) {
        out->data[i] = m->data[i];
    }
    return out;
}

Matrix* MatrixFlatten(Matrix* m) {
    Matrix* out = InitMatrix(1, m->rows*m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = MAT_AT(m, i, j);
        }
    }
    return out;
}

Matrix* MatrixClip(Matrix* m, double min, double max) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (MAT_AT(m, i, j) < min) {
                MAT_AT(out, i, j) = min;
            } else if (MAT_AT(m, i, j) > max) {
                MAT_AT(out, i, j) = max;
            } else {
                MAT_AT(out, i, j) = MAT_AT(m, i, j);
            }
        }
    }
    return out;
}

Matrix* MatrixSin(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = sin(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixCos(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = cos(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixTan(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = tan(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixArcSin(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = asin(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixArcCos(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = acos(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixArcTan(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = atan(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixSinh(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = sinh(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixCosh(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = cosh(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixTanh(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = tanh(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixArcSinh(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = asinh(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixArcCosh(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = acosh(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixArcTanh(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = atanh(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixCumSum(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    double sum = 0.0;

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            sum += MAT_AT(m, i, j);
            MAT_AT(out, i, j) = sum;
        }
    }
    return out;
}

Matrix* MatrixArange(double start, double end, double step) {
    int size = ceil((end - start) / step);
    Matrix* out = InitMatrix(1, size);
    double value = start;

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) = value;
            value += step;
        }
    }
    return out;
}

Matrix* MatrixLog(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = log(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixLog10(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = log10(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixLog2(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = log2(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixLog1p(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = log1p(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixReciprocal(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = (1 / MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixFabs(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = fabs(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixSqrt(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = sqrt(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixRSqrt(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = (1 / sqrt(MAT_AT(m, i, j)));
        }
    }
    return out;
}

double MatrixProd(Matrix* m) {
    double out = 1.0;

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out *= MAT_AT(m, i, j);
        }
    }
    return out;
}

Matrix* MatrixCumProd(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    double prod = 1.0;

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            prod *= MAT_AT(m, i, j);
            MAT_AT(out, i, j) = prod;
        }
    }
    return out;
}

Matrix* MatrixLerp(Matrix* m, Matrix* n, double weight) {
    assert(m->rows == n->rows);
    assert(m->cols == n->cols);

    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) = MAT_AT(m, i, j) + weight * (MAT_AT(n, i, j) - MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixNeg(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) =  -1 * MAT_AT(m, i, j);
        }
    }
    return out;
}

int MatrixNumel(Matrix* m) {
    return m->rows * m->cols;
}

Matrix* MatrixSign(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (MAT_AT(m, i, j) < 0) {
                MAT_AT(out, i, j) = -1.0;
            } else if (MAT_AT(m, i, j) > 0) {
                MAT_AT(out, i, j) = 1.0;
            } else {
                MAT_AT(out, i, j) = 0.0;
            }
        }
    }
    return out;
}

Matrix* MatrixEq(Matrix* m, Matrix* n) {
    assert(m->rows == n->rows);
    assert(m->cols == n->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) = (double)(MAT_AT(m, i, j) == MAT_AT(n, i, j));
        }
    }
    return out;
}

Matrix* MatrixLT(Matrix* m, Matrix* n) {
    assert(m->rows == n->rows);
    assert(m->cols == n->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) = (double)(MAT_AT(m, i, j) < MAT_AT(n, i, j));
        }
    }
    return out;
}

Matrix* MatrixGT(Matrix* m, Matrix* n) {
    assert(m->rows == n->rows);
    assert(m->cols == n->cols);
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            MAT_AT(out, i, j) = (double)(MAT_AT(m, i, j) > MAT_AT(n, i, j));
        }
    }
    return out;
}

Matrix* MatrixExp(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = exp(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixLogCumSumExp(Matrix* m) {
    return MatrixLog(MatrixCumSum(MatrixExp(m)));
}

Matrix* MatrixLGamma(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = log(gamma(MAT_AT(m, i, j)));
        }
    }
    return out;
}

void MatrixResize(Matrix* m, int rows, int cols) {
    m->rows = rows;
    m->cols = cols;
}

void MatrixResizeAs(Matrix* m, Matrix* n) {
    m->rows = n->rows;
    m->cols = n->cols;
}

void swap(double* a, double* b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

int parition(double arr[], int low, int high) {
    double pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i+1], &arr[high]);
    return (i + 1);
}

void quick_sort(double arr[], int low, int high) {
    if (low < high) {
        int pi = parition(arr, low, high);

        quick_sort(arr, low , pi-1);
        quick_sort(arr, pi+1, high);
    }
}

void MatrixSort(Matrix* m) {
    int size = m->rows * m->cols;
    quick_sort(m->data, 0, size - 1);
}

Matrix* MatrixArgSort(Matrix* m) {
    Matrix* m_copy = MatrixCopy(m);
    Matrix* out = InitMatrix(m->rows, m->cols);
    MatrixSort(m);

    for (int i = 0; i < m->rows*m->cols; i++) {
        for (int j = 0; j < m_copy->rows*m_copy->cols; j++) {
            if (m->data[i] == m_copy->data[j] && (out->data[i-1] != j)) {
                out->data[i] = j;
            }
        }
    }
    FreeMatrix(m_copy);
    return out;
}

Matrix* MatrixRepeat(Matrix* m, int rrows, int rcols) {
    int new_rows = m->rows * rrows;
    int new_cols = m->cols * rcols;
    Matrix* out = InitMatrix(new_rows, new_cols);

    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            int orig_i = i % m->rows;
            int orig_j = j % m->cols;
            MAT_AT(out, i, j) = MAT_AT(m, orig_i, orig_j);
        }
    }
    return out;
}

Matrix* MatrixTake(Matrix* m, Matrix* n) {
    Matrix* out = InitMatrix(n->rows, n->cols);

    for (int i = 0; i < n->rows; i++) {
        for (int j = 0; j < n->cols; j++) {
            MAT_AT(out, i, j) = m->data[(int)MAT_AT(n, i, j)];
        }
    }

    return out;
}

double randn() {
    static int use_last = 0;
    static double y2;
    double x1, x2, w, y1;

    if (use_last) {
        y1 = y2;
        use_last = 0;
    } else {
        do {
            x1 = 2.0 * rand() / (double)RAND_MAX - 1.0;
            x2 = 2.0 * rand() / (double)RAND_MAX - 1.0;
            w = x1*x1 + x2*x2;
        } while (w >= 1.0 || w == 0.0);

        w = sqrt((-2.0 * log(w)) / w);
        y1 = x1 * w;
        y2 = x2 * w;
        use_last = 1;
    }
    return y1;
}

Matrix* RandnMatrix(int rows, int cols, int seed) {
    if (seed != 0) {
        srand(seed);
    } else {
        srand(time(NULL));
    }

    Matrix* out = InitMatrix(rows, cols);
    if (out == NULL) return NULL;

    for (int i = 0; i < rows*cols; i++) {
        out->data[i] = randn();
    }
    return out;
}

void MatrixShape(Matrix* m) {
    printf("(%d, %d)\n", m->rows, m->cols);
}

Matrix* MatrixBroadcast(Matrix* m, int rows, int cols) {
    assert(m->rows*m->cols == 1);
    Matrix* out = MatrixFull(rows, cols, m->data[0]);
    return out;
}
