# Flash
my little linear algebra library

### Docs

#### Vector Functions
- <a href="#" onclick="Initialize a Vector of shape (1, size)">`Vector* InitVector(size_t size)`</a>
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
- `Vector* VectorOrthog(Vector* v)`
- `double VectorSum(Vector* v)`
- `Vector* VectorExp(Vector* v)`
- `bool VectorAllClose(Vector* v, Vector* w)`

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
- `MatrixTuple LUDecomposition(Matrix* A)`
- `double MatrixDeterminant(Matrix* m)`
- `double MatrixTrace(Matrix* m)`
- `double FrobeniusNorm(Matrix* m)`
- `double L1Norm(Matrix* m)`
- `double InfinityNorm(Matrix* m)`
- `double MatrixNorm(Matrix* m, char* type)`
- `Matrix* MatrixConcat(Matrix* m, Matrix* n, int axis)`
- `Matrix* MatrixCopy(Matrix* m)`
- `Matrix* MatrixNormalize(Matrix* m)`
- `void swap_rows(Matrix* m, int row1, int row2)`
- `void mult_row(Matrix* m, int row1, double scalar)`
- `void add_row(Matrix* m, int row1, int row2, double scalar)`
- `int find_pivot(Matrix* m, int col, int row)`
- `Matrix* MatrixRowEchelon(Matrix* m)`
- `Matrix* MatrixInverse(Matrix* m)`
- `MatrixTuple QRDecomposition(Matrix* m)`
- `Matrix* QRAlgorithm(Matrix* m)`
- `Vector* MatrixEig(Matrix* m)`
- `int non_zero_rows(Matrix* m)`
- `int MatrixRank(Matrix* m)`
- `Vector* MatrixDiagonal(Matrix* m, int k)`
- `Matrix* MatrixTril(Matrix* m, int diag)`
- `Matrix* MatrixTriu(Matrix* m, int diag)`
- `double MatrixMax(Matrix* mm)`
- `double MatrixMin(Matrix* m)`
- `double MatrixMean(Matrix* m)`
- `double MatrixStd(Matrix* m)`
- `Matrix* MatrixMaxVals(Matrix* m, int dim)`
- `Matrix* MatrixMinVals(Matrix* m, int dim)`
- `Matrix* MatrixMeanVals(Matrix* m, int dim)`
- `Matrix* MatrixStdVals(Matrix* m, int dim)`
- `bool MatrixAllClose(Matrix* m, Matrix* n)`
- `Matrix* MatrixSolve(Matrix* m, Matrix* n)`
- `Matrix* MatrixAbs(Matrix* m)` 
- `Matrix* CholeskyDecomposition(Matrix* m)`
- `SVDStruct SVD(Matrix* m)`
- `Matrix* MatrixEigVec(Matrix* m)`
- `Matrix* ToMatrix(Vector* v)`
- `Matrix* MatrixVectorMul(Matrix* m, Vector* v)`
- `Matrix* MatrixScalarAdd(Matrix* m, double x)`
- `Matrix* MatrixScalarSub(Matrix* m, double x)`
- `Matrix* MatrixScalarMul(Matrix* m, double x)`
- `Matrix* MatrixScalarDiv(Matrix* m, double x)`
- `Matrix* MatrixMultiply(Matrix* m, Matrix* n)`
- `double MatrixLogDeterminant(Matrix* m)`

#### TODO
- Cuda
- Python Frontend (maybe someday)
