# Flash
my little linear algebra library

## Docs

### Vector Functions
| Function | Description |
| `Vector* InitVector(size_t size)` | Initializes a Vector of shape (1, size) |
| `void FreeVector(Vector* v)` | Frees up the memory assigned to a `Vector*` | 
| `void VectorSetElements(Vector* v, double* values)` | Sets the data of the vector to the elements of an array |
| `void VectorSet(Vector* v, size_t index, double value)` | Sets the value of the element at the specified index in the vector |
| `double VectorGet(Vector* v, size_t index)` | Returns the value of the element at the specified index in the vector |
| `Vector* ZerosVector(size_t size)` | Creates a new vector of the specified size with all elements set to 0 |
| `Vector* OnesVector(size_t size)` | Creates a new vector of the specified size with all elements set to 1 |
| `Vector* RandomVector(size_t size, int seed)` | Creates a new vector of the specified size with random elements, using the provided seed for the random number generator |
| `void PrintVector(Vector* v)` | Prints the elements of the vector |
| `Vector* VectorAdd(Vector* v, Vector* w)` | Adds two vectors and returns the result as a new vector |
| `Vector* VectorSub(Vector* v, Vector* w)` | Subtracts two vectors and returns the result as a new vector |
| `Vector* VectorScale(Vector* v, int x)` | Multiplies a vector by a scalar value and returns the result as a new vector |
| `double VectorNorm(Vector* v)` | Calculates the norm (Euclidean length) of a vector |
| `double VectorDotProduct(Vector* v, Vector* w)` | Calculates the dot product of two vectors |
| `double VectorCrossProduct(Vector* v, Vector* w)` | Calculates the cross product of two vectors (assuming they are 3D vectors) |
| `double VectorAngle(Vector* v, Vector* w)` | Calculates the angle between two vectors |
| `bool VectorEqual(Vector* v, Vector* w)` | Checks if two vectors are equal (all elements are the same) |
| `Vector* VectorNormalize(Vector* v)` | Normalizes a vector (makes its norm equal to 1) and returns the result as a new vector |
| `Vector* VectorCopy(Vector* v)` | Creates a copy of the input vector |
| `Vector* VectorMultiply(Vector* v, Vector* w)` | Multiplies two vectors element-wise and returns the result as a new vector |
| `double VectorProjection(Vector* v, Vector* w)` | Calculates the projection of vector `v` onto vector `w` |
| `Vector* VectorTransform(Vector* v, Matrix* m)` | Applies a linear transformation represented by the matrix `m` to the vector `v`, and returns the result as a new vector |
| `double VectorSum(Vector* v)` | Calculates the sum of all elements in the vector |
| `Vector* VectorExp(Vector* v)` | Applies the exponential function to each element of the vector and returns the result as a new vector |
| `bool VectorAllClose(Vector* v, Vector* w)` | Checks if all elements in two vectors are close (within a small tolerance) to each other | 

### Matrix Functions
| Function | Description |
| `Matrix* InitMatrix(int rows, int cols)` | Initializes a matrix with the specified number of rows and columns |
| `void FreeMatrix(Matrix* m)` | Frees up the memory assigned to a `Matrix*` |
| `void SetElements(Matrix* m, double* values)` | Sets the data of the matrix to the elements of an array |
| `void PrintMatrix(Matrix* m)` | Prints the elements of the matrix |
| `Matrix* ZerosMatrix(int rows, int cols)` | Creates a new matrix of the specified size with all elements set to 0 |
| `Matrix* OnesMatrix(int rows, int cols)` | Creates a new matrix of the specified size with all elements set to 1 |
| `Matrix* IdentityMatrix(int side)` | Creates a new square identity matrix of the specified size |
| `Matrix* RandMatrix(int rows, int cols, int seed)` | Creates a new matrix of the specified size with random elements, using the provided seed for the random number generator |
| `Matrix* MatrixAdd(Matrix* m, Matrix* n)` | Adds two matrices and returns the result as a new matrix |
| `Matrix* MatrixSub(Matrix* m, Matrix* n)` | Subtracts two matrices and returns the result as a new matrix |
| `Matrix* MatrixScale(Matrix* m, int x)` | Multiplies a matrix by a scalar value and returns the result as a new matrix |
| `Matrix* MatrixTranspose(Matrix* m)` | Transposes a matrix and returns the result as a new matrix |
| `Matrix* MatrixMul(Matrix* m, Matrix* n)` | Multiplies two matrices and returns the result as a new matrix |
| `Matrix* MatrixSlice(Matrix* m, int from_rows, int to_rows, int from_cols, int to_cols)` | Slices a matrix and returns a new matrix containing the specified rows and columns |
| `double MatrixDeterminant(Matrix* m)` | Calculates the determinant of a matrix |
| `double MatrixLogDeterminant(Matrix* m)` | Calculates the natural logarithm of the determinant of a matrix |
| `double MatrixTrace(Matrix* m)` | Calculates the trace of a matrix (sum of diagonal elements) |
| `MatrixTuple LUDecomposition(Matrix* A)` | Computes the LU decomposition of a matrix |
| `double FrobeniusNorm(Matrix* m)` | Calculates the Frobenius norm of a matrix |
| `double L1Norm(Matrix* m)` | Calculates the L1 norm of a matrix |
| `double InfinityNorm(Matrix* m)` | Calculates the infinity norm of a matrix |
| `double MatrixNorm(Matrix* m, char* type)` | Calculates the norm of a matrix based on the specified type("frobenius"/"euclidean" - FrobeniusNorm, "l1", "infinity") |
| `Matrix* MatrixConcat(Matrix* m, Matrix* n, int axis)` | Concatenates two matrices along the specified axis and returns the result as a new matrix |
| `Matrix* MatrixCopy(Matrix* m)` | Creates a copy of the input matrix |
| `Matrix* MatrixNormalize(Matrix* m)` | Normalizes a matrix and returns the result as a new matrix |
| `void swap_rows(Matrix* m, int row1, int row2)` | Swaps two rows in a matrix |
| `void mult_row(Matrix* m, int row1, double scalar)` | Multiplies a row in a matrix by a scalar value |
| `void add_row(Matrix* m, int row1, int row2, double scalar)` | Adds a scalar multiple of one row to another row in a matrix |
| `int find_pivot(Matrix* m, int col, int row)` | Finds the pivot element in a matrix for row reduction |
| `Matrix* MatrixInverse(Matrix* m)` | Calculates the inverse of a matrix and returns the result as a new matrix |
| `Vector* MatrixEig(Matrix* m)` | Calculates the eigenvalues of a matrix and returns them as a vector |
| `int non_zero_rows(Matrix* m)` | Counts the number of non-zero rows in a matrix |
| `int MatrixRank(Matrix* m)` | Calculates the rank of a matrix |
| `Matrix* MatrixAbs(Matrix* m)` | Calculates the absolute value of each element in a matrix and returns the result as a new matrix |
| `double MatrixMax(Matrix* m)` | Calculates the maximum element in a matrix |
| `double MatrixMin(Matrix* m)` | Calculates the minimum element in a matrix |
| `double MatrixMean(Matrix* m)` | Calculates the mean value of all elements in a matrix |
| `double MatrixStd(Matrix* m)` | Calculates the standard deviation of all elements in a matrix |
| `Matrix* MatrixScalarAdd(Matrix* m, double x)` | Adds a scalar value to each element of a matrix and returns the result as a new matrix |
| `Matrix* MatrixScalarSub(Matrix* m, double x)` | Subtracts a scalar value from each element of a matrix and returns the result as a new matrix |
| `Matrix* MatrixScalarMul(Matrix* m, double x)` | Multiplies each element of a matrix by a scalar value and returns the result as a new matrix |
| `Matrix* MatrixScalarDiv(Matrix* m, double x)` | Divides each element of a matrix by a scalar value and returns the result as a new matrix|
| `Matrix* MatrixVectorMul(Matrix* m, Vector* v)` | Multiplies a matrix by a vector and returns the result as a new matrix |
| `Matrix* MatrixMultiply(Matrix* m, Matrix* n)` | Element-wise multiplies two matrices and returns the result as a new matrix |
| `Vector* MatrixDiagonal(Matrix* m, int k)` | Extracts the diagonal elements of a matrix at the specified offset and returns them as a vector |
| `Matrix* MatrixTril(Matrix* m, int diag)` | Extracts the lower triangular part of a matrix, including the specified diagonal, and returns it as a new matrix |
| `Matrix* MatrixTriu(Matrix* m, int diag)` | Extracts the upper triangular part of a matrix, including the specified diagonal, and returns it as a new matrix |
| `Matrix* MatrixMaxVals(Matrix* m, int dim)` | Calculates the maximum values along the specified dimension of a matrix and returns them as a new matrix |
| `Matrix* MatrixMinVals(Matrix* m, int dim)` | Calculates the minimum values along the specified dimension of a matrix and returns them as a new matrix |
| `Matrix* MatrixMeanVals(Matrix* m, int dim)` | Calculates the mean values along the specified dimension of a matrix and returns them as a new matrix |
| `Matrix* MatrixStdVals(Matrix* m, int dim)` | Calculates the standard deviation values along the specified dimension of a matrix and returns them as a new matrix |
| `bool MatrixAllClose(Matrix* m, Matrix* n)` | Checks if all elements in two matrices are close (within a small tolerance) to each other |
| `Matrix* MatrixSolve(Matrix* m, Matrix* n)` | Solves a system of linear equations represented by the matrices m and n and returns the solution as a new matrix |
| `Matrix* MatrixRowEchelon(Matrix* m)` | Converts a matrix to row echelon form and returns the result as a new matrix |
| `MatrixTuple QRDecomposition(Matrix* m)` | Computes the QR decomposition of a matrix |
| `Matrix* QRAlgorithm(Matrix* m)` | Applies the QR algorithm to a matrix and returns the result as a new matrix |
| `Matrix* CholeskyDecomposition(Matrix* m)` | Computes the Cholesky decomposition of a matrix |
| `SVDStruct SVD(Matrix* m)` | Computes the Singular Value Decomposition (SVD) of a matrix |
| `Matrix* MatrixEigVec(Matrix* m)` | Calculates the eigenvectors of a matrix and returns them as a new matrix |
| `Matrix* ToMatrix(Vector* v)` | Converts a vector to a matrix representation |

### Misc. Functions
| Function | Description |
| `double radian_to_degrees(double x)` | Converts an angle measure in radian to degrees. | 
| `void FreeMatrixTuple(MatrixTuple mt)` | Frees up the memory assigned to a `MatrixTuple` |
| `void FreeSVDStruct(SVDStruct svd)` | Frees up the memory assigned to a `SVDStruct` |

#### TODO
- Cuda
- Python Frontend (maybe someday)
