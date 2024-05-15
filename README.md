# Flash
my little linear algebra library

### Docs

#### Vector Functions
- `Vector* InitVector(size_t size)`
- `void FreeVector(Vector* v)`
- `void VectorSetElements(Vector* v, double* values)`
- `void VectorSet(Vector* v, size_t index, double value)`
- `double VectorGet(Vector* v, size_t index)`
- `void PrintVector(Vector* v)`
- `Vector* VectorAdd(Vector* v, Vector* w)`
- `Vector* VectorSub(Vector* v, Vector* w)`
- `Vector* VectorScale(Vector* v, int x)`
- `double VectorNorm(Vector* v)`
- `double VectorDotProduct(Vector* v, Vector* w)`
- `double VectorAngle(Vector* v, Vector* w)`
- `double VectorCrossProduct(Vector* v, Vector* w)`
- `bool VectorEqual(Vector* v, Vector* w)`
- `Vector* VectorNormalize(Vector* v)`
- `Vector* ZerosVector(size_t size)`
- `Vector* OnesVector(size_t size)`
- `Vector* RandomVector(size_t size, int seed)`
- `Vector* VectorCopy(Vector* v)`
- `Vector* VectorMultiply(Vector* v, Vector* w)`
- `double VectorProjection(Vector* v, Vector* w)`
- `Vector* VectorTransform(Vector* v, Matrix* m)`

#### Matrix Functions
- `Matrix* InitMatrix(int rows, int cols)`
- `void FreeMatrix(Matrix* m)`
- `void SetElements(Matrix* m, double* values)`
- `void PrintMatrix(Matrix* m)`
- `Matrix* RandMatrix(int rows, int cols, int seed)`
- `Matrix* MatrixAdd(Matrix* m, Matrix* n)`
- `Matrix* MatrixSub(Matrix* m, Matrix* n)`
- `Matrix* MatrixScale(Matrix* m, int x)`
- `Matrix* MatrixTranspose(Matrix* m)`
- `Matrix* OnesMatrix(int rows, int cols)`
- `Matrix* ZerosMatrix(int rows, int cols)`
- `Matrix* IdentityMatrix(int side)`
- `Matrix* MatrixMul(Matrix* m, Matrix* n)`
- `Matrix* MatrixSlice(Matrix* m, int from_rows, int to_rows, int from_cols, int to_cols)`
- `double MatrixDeterminant(Matrix* m)`
- `double MatrixTrace(Matrix* m)`
- `double FrobeniusNorm(Matrix* m)`
- `double L1Norm(Matrix* m)`
- `double InfinityNorm(Matrix* m)`
- `double MatrixNorm(Matrix* m, char* type)`
- `Matrix* MatrixConcat(Matrix* m, Matrix* n, int axis)`
- `Matrix* MatrixCopy(Matrix* m)`
- `Matrix* MatrixNormalize(Matrix* m)`
- `MatrixTuple LUDecomposition(Matrix* A)`
- `Matrix* MatrixRowEchelon(Matrix* m)`
- `Matrix* MatrixInverse(Matrix* m)`

#### TODO: 
- Matrix functions
    - [x] addition
    - [x] substract
    - [x] scalar multiplication
    - [x] matrix multiplication
    - [x] transpose
    - [x] inverse
    - [x] determinant
    - [x] trace
    - [ ] rank 
    - [ ] norm (frobenius, l1, l2, infinity)
    - [ ] factorization (LU, QR, Cholesky, SVD)
    - [ ] Eigenvectors and Eigenvalues 
    - [x] concatenation of matrices
    - [x] slice
    - [x] initialization (random, zeros, ones, identity)


