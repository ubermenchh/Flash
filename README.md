# Flash
<div align="center">
    <img src="./flash.png" width=400 height=400>
</div>

A little linear algebra library.

## Installation
```bash
# Clone the repo
git clone https://github.com/ubermenchh/Flash.git

# Compile the library
make 

# Install the library 
sudo make install 

# Cleanup the build files
make clean 

# Uninstall the library 
sudo make uninstall
```

## Example

### Initializing and Freeing a Matrix 
```c
int main() {
    Matrix* m = InitMatrix(4, 2); // Initializes a matrix
    double m_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    SetElements(m, m_data); // you can also do m->data = m_data;
    // NOTE: if you set the elements with m->data, make sure the number of elements 
    //       in the array are equal to product of rows and cols of the matrix.

    PrintMatrix(m); // prints the matrix

    FreeMatrix(m); // frees the memory allocated to the matrix. DO NOT FORGET TO FREE THE MATRIX.

    return 0;
}
```

```Output 
Matrix(data=(
[[  1.00000   2.00000  ]
 [  3.00000   4.00000  ]
 [  5.00000   6.00000  ]
 [  7.00000   8.00000  ]]
), size=(4, 2))
```

## Docs

### Vector Functions
```c
Vector* InitVector(size_t size)                                                         // Initialize a vector of length **size**.
void FreeVector(Vector* v)                                                              // Free the memory allocated for the vector.
void VectorSetElements(Vector* v, double* values)                                       // Set the elements of the vector to given values.
void VectorSet(Vector* v, size_t index, double value)                                   // Set the element at the specified index.
double VectorGet(Vector* v, size_t index)                                               // Get the element at the specified index.
void PrintVector(Vector* v)                                                             // Print the vector elements to standard output.
char* PrintVectorToString(Vector* v)                                                    // Return a string representation of the vector.
Vector* VectorAdd(Vector* v, Vector* w)                                                 // Add two vectors element-wise.
Vector* VectorSub(Vector* v, Vector* w)                                                 // Subtract two vectors element-wise.
Vector* VectorScale(Vector* v, int x)                                                   // Scale the vector by a scalar.
double VectorNorm(Vector* v)                                                            // Calculate the Euclidean norm of the vector.
double VectorDotProduct(Vector* v, Vector* w)                                           // Compute the dot product of two vectors.
double VectorAngle(Vector* v, Vector* w)                                                // Calculate the angle between two vectors.
double VectorCrossProduct(Vector* v, Vector* w)                                         // Compute the cross product of two vectors.
bool VectorEqual(Vector* v, Vector* w)                                                  // Check if two vectors are equal.
Vector* VectorNormalize(Vector* v)                                                      // Normalize the vector to have unit norm.
Vector* ZerosVector(size_t size)                                                        // Initialize a vector of zeros.
Vector* OnesVector(size_t size)                                                         // Initialize a vector of ones.
Vector* RandomVector(size_t size, int seed)                                             // Initialize a vector with random values.
Vector* VectorCopy(Vector* v)                                                           // Create a copy of the vector.
Vector* VectorMultiply(Vector* v, Vector* w)                                            // Multiply two vectors element-wise.
double VectorProjection(Vector* v, Vector* w)                                           // Compute the projection of v onto w.
Vector* VectorTransform(Vector* v, Matrix* m)                                           // Apply a matrix transformation to the vector.
Vector* VectorOrthog(Vector* v)                                                         // Return a vector orthogonal to the given vector.
double VectorSum(Vector* v)                                                             // Compute the sum of all elements in the vector.
Vector* VectorExp(Vector* v)                                                            // Apply the exponential function element-wise.
bool VectorAllClose(Vector* v, Vector* w)                                               // Check if two vectors are element-wise equal within a tolerance.
```

### Matrix Functions
```c 
Matrix* InitMatrix(int rows, int cols)                                                 // Initialize a matrix with specified rows and columns.
void FreeMatrix(Matrix* m)                                                             // Free the memory allocated for the matrix.
void SetElements(Matrix* m, double* values)                                            // Set the elements of the matrix with given values.
void PrintMatrix(Matrix* m)                                                            // Print the matrix elements to standard output.
Matrix* RandMatrix(int rows, int cols, int seed)                                       // Initialize a matrix with random values.
Matrix* RandnMatrix(int rows, int cols, int seed)                                      // Initialize a matrix with normally distributed values.
Matrix* MatrixAdd(Matrix* m, Matrix* n)                                                // Add two matrices element-wise.
Matrix* MatrixSub(Matrix* m, Matrix* n)                                                // Subtract two matrices element-wise.
Matrix* MatrixScale(Matrix* m, int x)                                                  // Scale the matrix by a scalar.
Matrix* MatrixTranspose(Matrix* m)                                                     // Transpose the matrix.
Matrix* OnesMatrix(int rows, int cols)                                                 // Initialize a matrix of ones.
Matrix* ZerosMatrix(int rows, int cols)                                                // Initialize a matrix of zeros.
Matrix* IdentityMatrix(int side)                                                       // Initialize an identity matrix.
Matrix* MatrixMul(Matrix* m, Matrix* n)                                                // Multiply two matrices.
Matrix* MatrixSlice(Matrix* m, int from_rows, int to_rows, int from_cols, int to_cols) // Slice the matrix.
MatrixTuple LUDecomposition(Matrix* A)                                                 // Perform LU decomposition.
double MatrixDeterminant(Matrix* m)                                                    // Calculate the determinant of the matrix.
double MatrixTrace(Matrix* m)                                                          // Calculate the trace of the matrix.
double FrobeniusNorm(Matrix* m)                                                        // Calculate the Frobenius norm.
double L1Norm(Matrix* m)                                                               // Calculate the L1 norm.
double InfinityNorm(Matrix* m)                                                         // Calculate the infinity norm.
double MatrixNorm(Matrix* m, char* type)                                               // Calculate the specified norm of the matrix.
Matrix* MatrixConcat(Matrix* m, Matrix* n, int axis)                                   // Concatenate two matrices along a specified axis.
Matrix* MatrixCopy(Matrix* m)                                                          // Create a copy of the matrix.
Matrix* MatrixNormalize(Matrix* m)                                                     // Normalize the matrix.
void swap_rows(Matrix* m, int row1, int row2)                                          // Swap two rows of the matrix.
void mult_row(Matrix* m, int row1, double scalar)                                      // Multiply a row by a scalar.
void add_row(Matrix* m, int row1, int row2, double scalar)                             // Add a multiple of one row to another.
int find_pivot(Matrix* m, int col, int row)                                            // Find the pivot element in a column.
Matrix* MatrixRowEchelon(Matrix* m)                                                    // Convert the matrix to row echelon form.
Matrix* MatrixInverse(Matrix* m)                                                       // Calculate the inverse of the matrix.
MatrixTuple QRDecomposition(Matrix* m)                                                 // Perform QR decomposition.
Matrix* QRAlgorithm(Matrix* m)                                                         // Apply the QR algorithm to the matrix.
Vector* MatrixEig(Matrix* m)                                                           // Compute the eigenvalues of the matrix.
int non_zero_rows(Matrix* m)                                                           // Count the non-zero rows.
int MatrixRank(Matrix* m)                                                              // Calculate the rank of the matrix.
Vector* MatrixDiagonal(Matrix* m, int k)                                               // Extract the k-th diagonal.
Matrix* MatrixTril(Matrix* m, int diag)                                                // Extract the lower triangular part.
Matrix* MatrixTriu(Matrix* m, int diag)                                                // Extract the upper triangular part.
double MatrixSum(Matrix* m)                                                            // Calculate the sum of all elements.
double MatrixMax(Matrix* m)                                                            // Find the maximum element.
double MatrixMin(Matrix* m)                                                            // Find the minimum element.
double MatrixMean(Matrix* m)                                                           // Calculate the mean of the elements.
double MatrixStd(Matrix* m)                                                            // Calculate the standard deviation.
Matrix* MatrixSumVals(Matrix* m, int dim)                                              // Sum elements along a dimension.
Matrix* MatrixMaxVals(Matrix* m, int dim)                                              // Max elements along a dimension.
Matrix* MatrixMinVals(Matrix* m, int dim)                                              // Min elements along a dimension.
Matrix* MatrixMeanVals(Matrix* m, int dim)                                             // Mean elements along a dimension.
Matrix* MatrixStdVals(Matrix* m, int dim)                                              // Standard deviation along a dimension.
bool MatrixAllClose(Matrix* m, Matrix* n)                                              // Check if two matrices are element-wise close.
Matrix* MatrixSolve(Matrix* m, Matrix* n)                                              // Solve a linear system.
Matrix* MatrixAbs(Matrix* m)                                                           // Apply absolute value element-wise.
Matrix* CholeskyDecomposition(Matrix* m)                                               // Perform Cholesky decomposition.
SVDStruct SVD(Matrix* m)                                                               // Perform Singular Value Decomposition.
Matrix* MatrixEigVec(Matrix* m)                                                        // Compute the eigenvectors.
Matrix* ToMatrix(Vector* v)                                                            // Convert a vector to a matrix.
Matrix* MatrixVectorMul(Matrix* m, Vector* v)                                          // Multiply a matrix by a vector.
Matrix* MatrixScalarAdd(Matrix* m, double x)                                           // Add a scalar to the matrix.
Matrix* MatrixScalarSub(Matrix* m, double x)                                           // Subtract a scalar from the matrix.
Matrix* MatrixScalarMul(Matrix* m, double x)                                           // Multiply the matrix by a scalar.
Matrix* MatrixScalarDiv(Matrix* m, double x)                                           // Divide the matrix by a scalar.
Matrix* MatrixMultiply(Matrix* m, Matrix* n)                                           // Multiply two matrices.
double MatrixLogDeterminant(Matrix* m)                                                 // Calculate the log determinant.
Matrix* MatrixPower(Matrix* m, double exp)                                             // Raise the matrix to a power.
Matrix* MatrixOnesLike(Matrix* m)                                                      // Initialize a matrix of ones with same shape.
Matrix* MatrixZerosLike(Matrix* m)                                                     // Initialize a matrix of zeros with same shape.
Matrix* MatrixFull(int rows, int cols, double value)                                   // Initialize a matrix with a specific value.
Matrix* MatrixFullLike(Matrix* m, double value)                                        // Initialize a matrix with specific value, same shape.
void MatrixFill(Matrix* m, double value)                                               // Fill the matrix with a specific value.
Matrix* MatrixReshape(Matrix* m, int rows, int cols)                                   // Reshape the matrix.
Matrix* MatrixFlatten(Matrix* m)                                                       // Flatten the matrix.
Matrix* MatrixClip(Matrix* m, double min, double max)                                  // Clip the matrix elements between min and max.
Matrix* MatrixSin(Matrix* m)                                                           // Apply sine function element-wise.
Matrix* MatrixCos(Matrix* m)                                                           // Apply cosine function element-wise.
Matrix* MatrixTan(Matrix* m)                                                           // Apply tangent function element-wise.
Matrix* MatrixArcSin(Matrix* m)                                                        // Apply arcsine function element-wise.
Matrix* MatrixArcCos(Matrix* m)                                                        // Apply arccosine function element-wise.
Matrix* MatrixArcTan(Matrix* m)                                                        // Apply arctangent function element-wise.
Matrix* MatrixSinh(Matrix* m)                                                          // Apply hyperbolic sine function element-wise.
Matrix* MatrixCosh(Matrix* m)                                                          // Apply hyperbolic cosine function element-wise.
Matrix* MatrixTanh(Matrix* m)                                                          // Apply hyperbolic tangent function element-wise.
Matrix* MatrixArcSinh(Matrix* m)                                                       // Apply inverse hyperbolic sine function element-wise.
Matrix* MatrixArcCosh(Matrix* m)                                                       // Apply inverse hyperbolic cosine function element-wise.
Matrix* MatrixArcTanh(Matrix* m)                                                       // Apply inverse hyperbolic tangent function element-wise.
Matrix* MatrixCumSum(Matrix* m)                                                        // Compute cumulative sum of elements.
Matrix* MatrixArange(double start, double end, double step)                            // Create a range of values.
Matrix* MatrixLog(Matrix* m)                                                           // Apply natural logarithm element-wise.
Matrix* MatrixLog10(Matrix* m)                                                         // Apply base-10 logarithm element-wise.
Matrix* MatrixLog2(Matrix* m)                                                          // Apply base-2 logarithm element-wise.
Matrix* MatrixLog1p(Matrix* m)                                                         // Apply log(1 + x) element-wise.
Matrix* MatrixReciprocal(Matrix* m)                                                    // Apply reciprocal function element-wise.
Matrix* MatrixFabs(Matrix* m)                                                          // Apply absolute value function element-wise.
Matrix* MatrixSqrt(Matrix* m)                                                          // Apply square root function element-wise.
Matrix* MatrixRSqrt(Matrix* m)                                                         // Apply reciprocal of square root function element-wise.
double MatrixProd(Matrix* m)                                                           // Compute the product of all elements.
Matrix* MatrixCumProd(Matrix* m)                                                       // Compute cumulative product of elements.
Matrix* MatrixLerp(Matrix* m, Matrix* n, double weight)                                // Linear interpolation between two matrices.
Matrix* MatrixNeg(Matrix* m)                                                           // Negate all elements in the matrix.
int MatrixNumel(Matrix* m)                                                             // Get the number of elements in the matrix.
Matrix* MatrixSign(Matrix* m)                                                          // Apply sign function element-wise.
Matrix* MatrixEq(Matrix* m, Matrix* n)                                                 // Check element-wise equality.
Matrix* MatrixLT(Matrix* m, Matrix* n)                                                 // Check element-wise less than.
Matrix* MatrixGT(Matrix* m, Matrix* n)                                                 // Check element-wise greater than.
Matrix* MatrixExp(Matrix* m)                                                           // Apply exponential function element-wise.
Matrix* MatrixLogSumExp(Matrix* m)                                                     // Compute log-sum-exp over elements.
Matrix* MatrixLGamma(Matrix* m)                                                        // Apply log-gamma function element-wise.
void MatrixResize(Matrix* m, int rows, int cols)                                       // Resize the matrix.
void MatrixResizeAs(Matrix* m, Matrix* n)                                              // Resize the matrix to match another matrix.
void MatrixSort(Matrix* m)                                                             // Sort elements in ascending order.
Matrix* MatrixArgSort(Matrix* m)                                                       // Return indices that would sort the matrix.
Matrix* MatrixRepeat(Matrix* m, int rrows, int rcols)                                  // Repeat the matrix along rows and columns.
Matrix* MatrixTake(Matrix* m, Matrix* n)                                               // Select elements from the matrix.
int MatrixArgMax(Matrix* m)                                                            // Find the index of the maximum element.
int MatrixArgMin(Matrix* m)                                                            // Find the index of the minimum element.
Matrix* MatrixArgMaxVals(Matrix* m, int dim)                                           // Indices of the max values along a dimension.
Matrix* MatrixArgMinVals(Matrix* m, int dim)                                           // Indices of the min values along a dimension.
```

### Misc. Functions
```c
double radian_to_degrees(double x)                                                      // Converts an angle measure in radian to degrees.
void FreeMatrixTuple(MatrixTuple mt)                                                    // Frees up the memory assigned to a MatrixTuple.
void FreeSVDStruct(SVDStruct svd)                                                       // Frees up the memory assigned to a SVDStruct.
``` 

#### TODO
- Cuda
- Python Frontend (maybe someday)
