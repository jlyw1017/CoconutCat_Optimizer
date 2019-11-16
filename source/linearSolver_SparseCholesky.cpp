#include "linearSolver_SparseCholesky.h"

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

template<typename  T>
void deepCopyArray(T* targetArray, T* srcArray, int m, int nnz,int target){
    if (target == 0){
        for(int i = 0; i < nnz; i++) {
            targetArray[i] = srcArray[i]+1;
        }
    }
    else if(target == 1){
        for(int i = 0; i < m; i++) {
            targetArray[i] = srcArray[i]+1;
        }
        targetArray[m] = nnz+1;
    }
    else if(target == 2){
        for(int i = 0; i < nnz; i++) {
            targetArray[i] = srcArray[i];
        }
    }
}

Eigen::VectorXd cuda_Solver(Eigen::SparseMatrix<double,Eigen::RowMajor> S,Eigen::VectorXd b_) //
{
    //double* csrVal, int* csrRowPtr, int* csrColInd, int nnzInput, int mInput
    // GPU does batch QR
    csrqrInfo_t info = NULL;
    // Set Status for Debugg
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = NULL; // working space for numerical factorization

    // Set Host Variables
    const int m = S.outerSize(); // row number
    const int m_plusone = m+1;
    const int nnzA = S.nonZeros();// nozero values number 非零元素个数
    int csrColIndA[nnzA];
    int csrRowPtrA[m_plusone];
    double csrValA[nnzA];
    deepCopyArray(&csrColIndA[0],(int*)S.innerIndexPtr(),m,nnzA,0);
    deepCopyArray(&csrRowPtrA[0],(int*)S.outerIndexPtr(),m,nnzA,1);
    deepCopyArray(&csrValA[0],S.valuePtr(),m,nnzA,2);
    double b[m];
    for(int i=0;i<m;i++){
        b[i] = b_(i);
    }
    double x[m] = {0.0, 0.0 ,0.0, 0.0, 0.0};
/*
    std::cout << "csrColIndA:" << std::endl;
    for(int i=0;i<nnzA;i++) {
        std::cout  << csrColIndA[i] << " ";
    }
    std::cout << std::endl << "csrRowPtrA:" << std::endl;
    for(int i=0;i<m+1;i++){
        std::cout  << csrRowPtrA[i] << " ";
    }

    std::cout << std::endl << "csrValA:" << std::endl;
    for(int i=0;i<nnzA;i++) {
        std::cout << csrValA[i] << " ";
    }
*/

/*
    const int m = 4 ; // row number
    const int nnzA = 7;// nozero values number 非零元素个数
    // integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.
    int csrRowPtrA[m+1]  = { 1, 2, 3, 4, 8};  // Why m+1? //
    int csrColIndA[nnzA] = { 1, 2, 3, 1, 2, 3, 4};
    double csrValA[nnzA] = { 1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 4.0};
    double x[m] = {0.0, 0.0 ,0.0, 0.0};
    const double b[m] = {1.0, 1.0, 1.0, 1.0};
*/

    /*
    const int m = 2 ; // row number
    const int nnzA = 2;// nozero values number 非零元素个数
    const int csrRowPtrA[m+1]  = { 1, 2, 3};  // Why m+1? //
    const int csrColIndA[nnzA] = { 1, 2};
    const double csrValA[nnzA] = { 1.0, 1.0};
    const double b[m] = {1.0, 2.0};
    double x[m] = {0.0, 0.0};
    */

    int singularity;
    // std::cout << "Host Variable Setted!" << std::endl;

    // step 2: create cusolver handle and matrix descriptor
    // set Solver
    cusolverSpHandle_t cusolverH = NULL;
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // Set Mat
    // cusparseMatDescr_t describe the shape and properties of a matrix.
    // create and set cusparseMatDescr_t here
    cusparseMatDescr_t descrA = NULL;
    cusparse_status = cusparseCreateMatDescr(&descrA);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); // base-1
    // Set Info
    cusolver_status = cusolverSpCreateCsrqrInfo(&info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // Set GPU Variables
    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;
    double *d_b = NULL; // m
    double *d_x = NULL; // m
    int * d_singularity = &singularity;

    // allocate memory on CUDA
    cudaStat1 = cudaMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA );
    cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA);
    cudaStat3 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * m_plusone);
    cudaStat4 = cudaMalloc ((void**)&d_b         , sizeof(double) * m );
    cudaStat5 = cudaMalloc ((void**)&d_x         , sizeof(double) * m );
    //cudaStat6 = cudaMalloc ((void**)&d_singularity , sizeof(int));
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);
    assert(cudaStat6 == cudaSuccess);

    // std::cout << "GPU Variable Setted!" << std::endl;

    // Copy to GPU
    cudaStat1 = cudaMemcpy(d_csrValA   , csrValA, sizeof(double) * nnzA , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * m_plusone, cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(d_b, b, sizeof(double) * m , cudaMemcpyHostToDevice);
    //cudaStat5 = cudaMemcpy(d_x, x, sizeof(double) * m , cudaMemcpyHostToDevice);
    //cudaStat6 = cudaMemcpy(d_singularity, singularity, sizeof(int) , cudaMemcpyHostToDevice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);
    assert(cudaStat6 == cudaSuccess);
    // std::cout << "Copy to GPU successful!" << std::endl;
    if (cudaStat1 != cudaSuccess) {
        fprintf(stderr, "Failed %s\n", cudaGetErrorString(cudaStat1));
    }
    if (cudaStat2 != cudaSuccess) {
        fprintf(stderr, "Failed %s\n", cudaGetErrorString(cudaStat2));
    }
    if (cudaStat3 != cudaSuccess) {
        fprintf(stderr, "Failed %s\n", cudaGetErrorString(cudaStat3));
    }
    if (cudaStat4 != cudaSuccess) {
        fprintf(stderr, "Failed %s\n", cudaGetErrorString(cudaStat4));
    }
    if (cudaStat5 != cudaSuccess) {
        fprintf(stderr, "Failed %s\n", cudaGetErrorString(cudaStat5));
    }
    // Calculate the Linear Function
    // assume device memory is big enough to compute all matrices.
    double tolerance = 0.001 ;
    int reorder = 0;
    cusolver_status = cusolverSpDcsrlsvchol(
        cusolverH,
        m,
        nnzA,
        descrA,
        d_csrValA,
        d_csrRowPtrA,
        d_csrColIndA,
        d_b,
        tolerance,
        reorder,
        d_x,
        d_singularity);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    // std::cout << "Solved!" << std::endl;

// check residual
    cudaStat1 = cudaMemcpy(x, d_x, sizeof(double)*m, cudaMemcpyDeviceToHost);
    if (cudaStat1 != cudaSuccess) {
        fprintf(stderr, "Failed %s\n", cudaGetErrorString(cudaStat1));
    }
    /*
    std::cout << "Result:" << std::endl;
    std::cout << x[0] << std::endl;
    std::cout << x[1] << std::endl;
    std::cout << x[2] << std::endl;
    std::cout << x[3] << std::endl;
    std::cout << x[4] << std::endl;
     */
    Eigen::VectorXd result(m);
    for(int i=0;i<m;i++){
        result[i] = x[i];
    }
    return result;
}


