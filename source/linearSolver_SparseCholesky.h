#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cusolverSp.h>
#include <cuda_runtime_api.h>

template<typename  T>
void deepCopyArray(T* targetArray, T* srcArray, int m, int nnz,int target);
Eigen::VectorXd cuda_Solver(Eigen::SparseMatrix<double,Eigen::RowMajor> S,Eigen::VectorXd b_);


// TEST
/*      | 1                |
 *  A = |       2          |
 *      |            3     |
 *      | 0.1  0.1  0.1  4 |
 *  CSR of A is based-1
 *
 *  b = [1 1 1 1]
 */

// GPU does Cholesky
// d_A is CSR format, d_csrValA is of size nnzA   nnzA（nozero values of A)
// CSR 格式存储，其中VAL存贮值，ColIndA存储每行值在哪列，RowPtr存储每行开头在ColndA的第几位
// Specially RowPtr contains
// integer array of m + 1 elements that contains the start of every row
// and the end of the last row plus one.
// d_x is a matrix of size  m
// d_b is a matrix of size  m



