#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define MAT_AT(m, i, j) (m)->data[(i) * (m)->cols + (j)]
#define ATOL 1e-08
#define RTOL 1e-05

typedef struct {
    size_t size;
    double* data;
} Vector;

typedef struct {
    int rows; 
    int cols;
    double* data;
} Matrix;

typedef struct {
    Matrix* first;
    Matrix* second;
} MatrixTuple;

typedef struct {
    Matrix* U;
    Vector* S;
    Matrix* V;
} SVDStruct;


double radian_to_degrees(double x);
void FreeMatrixTuple(MatrixTuple mt);
void FreeSVDStruct(SVDStruct svd);

Vector* InitVector(size_t size);
void FreeVector(Vector* v);
void VectorSetElements(Vector* v, double* values);
void VectorSet(Vector* v, size_t index, double value);
double VectorGet(Vector* v, size_t index);
void PrintVector(Vector* v);
char* PrintVectorToString(Vector* v);
Vector* VectorAdd(Vector* v, Vector* w);
Vector* VectorSub(Vector* v, Vector* w);
Vector* VectorScale(Vector* v, int x);
double VectorNorm(Vector* v);
double VectorDotProduct(Vector* v, Vector* w);
double VectorAngle(Vector* v, Vector* w);
double VectorCrossProduct(Vector* v, Vector* w);
bool VectorEqual(Vector* v, Vector* w);
Vector* VectorNormalize(Vector* v);
Vector* ZerosVector(size_t size);
Vector* OnesVector(size_t size);
Vector* RandomVector(size_t size, int seed);
Vector* VectorCopy(Vector* v);
Vector* VectorMultiply(Vector* v, Vector* w);
double VectorProjection(Vector* v, Vector* w);
Vector* VectorTransform(Vector* v, Matrix* m);
Vector* VectorOrthog(Vector* v);
double VectorSum(Vector* v);
Vector* VectorExp(Vector* v);
bool VectorAllClose(Vector* v, Vector* w);

Matrix* InitMatrix(int rows, int cols);
void FreeMatrix(Matrix* m);
void SetElements(Matrix* m, double* values);
void PrintMatrix(Matrix* m);
Matrix* RandMatrix(int rows, int cols, int seed);
Matrix* MatrixAdd(Matrix* m, Matrix* n);
Matrix* MatrixSub(Matrix* m, Matrix* n);
Matrix* MatrixScale(Matrix* m, int x);
Matrix* MatrixTranspose(Matrix* m);
Matrix* OnesMatrix(int rows, int cols);
Matrix* ZerosMatrix(int rows, int cols);
Matrix* MatrixMask(int rows, int cols, double prob);
Matrix* IdentityMatrix(int side);
Matrix* MatrixMul(Matrix* m, Matrix* n);
Matrix* MatrixSlice(Matrix* m, int from_rows, int to_rows, int from_cols, int to_cols);
MatrixTuple LUDecomposition(Matrix* A);
double MatrixDeterminant(Matrix* m);
double MatrixTrace(Matrix* m);
double FrobeniusNorm(Matrix* m);
double L1Norm(Matrix* m);
double InfinityNorm(Matrix* m);
double MatrixNorm(Matrix* m, char* type);
Matrix* MatrixConcat(Matrix* m, Matrix* n, int axis);
Matrix* MatrixCopy(Matrix* m);
Matrix* MatrixNormalize(Matrix* m);
void swap_rows(Matrix* m, int row1, int row2);
void mult_row(Matrix* m, int row1, double scalar);
void add_row(Matrix* m, int row1, int row2, double scalar);
int find_pivot(Matrix* m, int col, int row);
Matrix* MatrixRowEchelon(Matrix* m);
Matrix* MatrixInverse(Matrix* m);
MatrixTuple QRDecomposition(Matrix* m);
Matrix* QRAlgorithm(Matrix* m);
Vector* MatrixEig(Matrix* m);
int non_zero_rows(Matrix* m);
int MatrixRank(Matrix* m);
Vector* MatrixDiagonal(Matrix* m, int k);
Matrix* MatrixTril(Matrix* m, int diag);
Matrix* MatrixTriu(Matrix* m, int diag);
double MatrixSum(Matrix* m);
double MatrixMax(Matrix* m);
double MatrixMin(Matrix* m);
double MatrixMean(Matrix* m);
double MatrixStd(Matrix* m);
Matrix* MatrixSumVals(Matrix* m, int dim);
Matrix* MatrixMaxVals(Matrix* m, int dim);
Matrix* MatrixMinVals(Matrix* m, int dim);
Matrix* MatrixMeanVals(Matrix* m, int dim);
Matrix* MatrixStdVals(Matrix* m, int dim);
bool MatrixAllClose(Matrix* m, Matrix* n);
Matrix* MatrixSolve(Matrix* m, Matrix* n);
Matrix* MatrixAbs(Matrix* m);
Matrix* CholeskyDecomposition(Matrix* m);
SVDStruct SVD(Matrix* m);
Matrix* MatrixEigVec(Matrix* m);
Matrix* ToMatrix(Vector* v);
Matrix* MatrixVectorMul(Matrix* m, Vector* v);
Matrix* MatrixScalarAdd(Matrix* m, double x);
Matrix* MatrixScalarSub(Matrix* m, double x);
Matrix* MatrixScalarMul(Matrix* m, double x);
Matrix* MatrixScalarDiv(Matrix* m, double x);
Matrix* MatrixMultiply(Matrix* m, Matrix* n);
Matrix* MatrixDivide(Matrix* m, Matrix* n);
double MatrixLogDeterminant(Matrix* m);
Matrix* MatrixPower(Matrix* m, double exp);
Matrix* MatrixOnesLike(Matrix* m);
Matrix* MatrixZerosLike(Matrix* m);
Matrix* MatrixFull(int rows, int cols, double value);
Matrix* MatrixFullLike(Matrix* m, double value);
void MatrixFill(Matrix* m, double value);
Matrix* MatrixReshape(Matrix* m, int rows, int cols);
Matrix* MatrixFlatten(Matrix* m);
Matrix* MatrixClip(Matrix* m, double min, double max);
Matrix* MatrixSin(Matrix* m);
Matrix* MatrixCos(Matrix* m);
Matrix* MatrixTan(Matrix* m);
Matrix* MatrixArcSin(Matrix* m);
Matrix* MatrixArcCos(Matrix* m);
Matrix* MatrixArcTan(Matrix* m);
Matrix* MatrixSinh(Matrix* m);
Matrix* MatrixCosh(Matrix* m);
Matrix* MatrixTanh(Matrix* m);
Matrix* MatrixArcSinh(Matrix* m);
Matrix* MatrixArcCosh(Matrix* m);
Matrix* MatrixArcTanh(Matrix* m);
Matrix* MatrixCumSum(Matrix* m);
Matrix* MatrixArange(double start, double end, double step);
Matrix* MatrixLog(Matrix* m);
Matrix* MatrixLog10(Matrix* m);
Matrix* MatrixLog2(Matrix* m);
Matrix* MatrixLog1p(Matrix* m);
Matrix* MatrixReciprocal(Matrix* m);
Matrix* MatrixFabs(Matrix* m);
Matrix* MatrixSqrt(Matrix* m);
Matrix* MatrixRSqrt(Matrix* m);
double MatrixProd(Matrix* m);
Matrix* MatrixCumProd(Matrix* m);
Matrix* MatrixLerp(Matrix* m, Matrix* n, double weight);
Matrix* MatrixNeg(Matrix* m);
int MatrixNumel(Matrix* m);
Matrix* MatrixSign(Matrix* m);
Matrix* MatrixEq(Matrix* m, Matrix* n);
Matrix* MatrixLT(Matrix* m, Matrix* n);
Matrix* MatrixGT(Matrix* m, Matrix* n);
Matrix* MatrixExp(Matrix* m);
Matrix* MatrixLogSumExp(Matrix* m);
Matrix* MatrixLGamma(Matrix* m);
void MatrixResize(Matrix* m, int rows, int cols);
void MatrixResizeAs(Matrix* m, Matrix* n);
void MatrixSort(Matrix* m);
Matrix* MatrixArgSort(Matrix* m);
Matrix* MatrixRepeat(Matrix* m, int rrows, int rcols);
Matrix* MatrixTake(Matrix* m, Matrix* n);
int MatrixArgMax(Matrix* m);
int MatrixArgMin(Matrix* m);
Matrix* MatrixArgMaxVals(Matrix* m, int dim);
Matrix* MatrixArgMinVals(Matrix* m, int dim);
Matrix* RandnMatrix(int rows, int cols, int seed);
void MatrixShape(Matrix* m);
