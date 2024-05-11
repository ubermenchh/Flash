# Flash
my little linear algebra library

### Docs

#### 2D Vector Functions
- `Vector2d Add2d(Vector2d v, Vector2d w)`
- `Vector2d Subtract2d(Vector2d v, Vector2d w)`
- `Vector2d Scale2d(Vector2d v, int x)`
- `Vector2d DivideScalar2d(Vector2d v, int x)`
- `void Print2d(Vector2d v)`
- `float Norm2d(Vector2d v)`
- `float DotProduct2d(Vector2d v, Vector2d w)`
- `float Angle2d(Vector2d v, Vector2d w)`
- `float CrossProduct2d(Vector2d v, Vector2d w)`
- `bool Equal2d(Vector2d v, Vector2d w)`
- `Vector2d Normalize2d(Vector2d v)`
- `Vector2d Zeros2d(void)`
- `Vector2d Ones2d(void)`
- `Vector2d Init2d(int seed)`
- `Vector2d Copy2d(Vector2d v)`
- `Vector2d Multiply2d(Vector2d v, Vector2d w)`

#### 3D Vector Functions
- `Vector3d Add3d(Vector3d v, Vector3d w)`
- `Vector3d Subtract3d(Vector3d v, Vector3d w)`
- `Vector3d Scale3d(Vector3d v, int x)`
- `Vector3d DivideScalar3d(Vector3d v, int x)`
- `void Print3d(Vector3d v)`
- `float Norm3d(Vector3d v)`
- `float DotProduct3d(Vector3d v, Vector3d w)`
- `float Angle3d(Vector3d v, Vector3d w)`
- `float CrossProduct3d(Vector3d v, Vector3d w)`
- `bool Equal3d(Vector3d v, Vector3d w)`
- `Vector3d Normalize3d(Vector3d v)`
- `Vector3d Zeros3d(void)`
- `Vector3d Ones3d(void)`
- `Vector3d Init3d(int seed)`
- `Vector3d Copy3d(Vector3d v)`
- `Vector3d Multiply3d(Vector3d v, Vector3d w)`

#### Matrix Functions
- `Matrix* InitMatrix(int rows, int cols);`
- `void FreeMatrix(Matrix* m);`
- `void SetElements(Matrix* m, double* values);`
- `void PrintMatrix(Matrix* m);`
- `Matrix* RandMatrix(int rows, int cols, int seed);`
- `Matrix* matadd(Matrix* m, Matrix* n);`
- `Matrix* matsub(Matrix* m, Matrix* n);`
- `Matrix* scalarmul(Matrix* m, int x);`
- `Matrix* transpose(Matrix* m);`
- `Matrix* ones(int rows, int cols);`
- `Matrix* zeros(int rows, int cols);`
- `Matrix* identity(int side);`
- `Matrix* matmul(Matrix* m, Matrix* n);`
- `Matrix* slice(Matrix* m, int from_rows, int to_rows, int from_cols, int to_cols);`


#### TODO: 
- Matrix functions
    - [x] addition
    - [x] substract
    - [x] scalar multiplication
    - [x] matrix multiplication
    - [x] transpose
    - [ ] inverse
    - [ ] determinant
    - [ ] trace
    - [ ] rank 
    - [ ] norm (frobenius, l1, l2, infinity)
    - [ ] factorization (LU, QR, Cholesky, SVD)
    - [ ] Eigenvectors and Eigenvalues 
    - [ ] concatenation of matrices
    - [x] slice
    - [x] initialization (random, zeros, ones, identity)
- might get rid of seperate 2d and 3d functions and make it dynamic (idk how but we'll see)


