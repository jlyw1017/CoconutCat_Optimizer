#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

void cuda_Solver(){
    // Set Solver
    cusolverDnHandle_t handle = NULL;
    cudaStream_t stream = NULL;

    // Set Satus for Error checking
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;


    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    const int batchSize = 2;
    const int nrhs = 1;
    const int m = 3;
    const int lda = m;
    const int ldb = m;
    /*
     *      | 1     2     3 |
     * A0 = | 2     5     5 | = L0 * L0**T
     *      | 3     5    12 |
     *
     *            | 1.0000         0         0 |
     * where L0 = | 2.0000    1.0000         0 |
     *            | 3.0000   -1.0000    1.4142 |
     *
     *      | 1     2     3 |
     * A1 = | 2     4     5 | is not s.p.d., failed at row 2
     *      | 3     5    12 |
     *
     */

    double A0[lda*m] = { 1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0 };
    double A1[lda*m] = { 1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 12.0 };
    double B0[m] = { 1.0, 1.0, 1.0 };
    double X0[m]; /* X0 = A0\B0 */
    int infoArray[batchSize]; /* host copy of error info */

    double L0[lda*m]; /* cholesky factor of A0 */

    double *Aarray[batchSize];
    double *Barray[batchSize];

    double **d_Aarray = NULL;
    double **d_Barray = NULL;
    int *d_infoArray = NULL;

    printf("example of batched Cholesky \n");

    printf("A0 = (matlab base-1)\n");
    printMatrix(m, m, A0, lda, "A0");
    printf("=====\n");

    printf("A1 = (matlab base-1)\n");
    printMatrix(m, m, A1, lda, "A1");
    printf("=====\n");

    printf("B0 = (matlab base-1)\n");
    printMatrix(m, 1, B0, ldb, "B0");
    printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&handle);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(handle, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* step 2: copy A to device */
    for(int j = 0 ; j < batchSize ; j++){
    cudaStat1 = cudaMalloc ((void**)&Aarray[j], sizeof(double) * lda * m);
    assert(cudaSuccess == cudaStat1);
    cudaStat2 = cudaMalloc ((void**)&Barray[j], sizeof(double) * ldb * nrhs);
    assert(cudaSuccess == cudaStat2);
    }
    cudaStat1 = cudaMalloc ((void**)&d_infoArray, sizeof(int)*batchSize);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMalloc ((void**)&d_Aarray, sizeof(double*) * batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_Barray, sizeof(double*) * batchSize);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    cudaStat1 = cudaMemcpy(Aarray[0], A0, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(Aarray[1], A1, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);


    cudaStat1 = cudaMemcpy(Barray[0], B0, sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(Barray[1], B0, sizeof(double) * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    cudaStat1 = cudaMemcpy(d_Aarray, Aarray, sizeof(double*)*batchSize, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_Barray, Barray, sizeof(double*)*batchSize, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    cudaDeviceSynchronize();

    /* step 3: Cholesky factorization */
    status = cusolverDnDpotrfBatched(
            handle,
            uplo,
            m,
            d_Aarray,
            lda,
            d_infoArray,
            batchSize);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(infoArray, d_infoArray, sizeof(int)*batchSize, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(L0, Aarray[0], sizeof(double) * lda * m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    for(int j = 0 ; j < batchSize ; j++){
    printf("info[%d] = %d\n", j, infoArray[j]);
    }

    assert( 0 == infoArray[0] );
    /* A1 is singular */
    assert( 2 == infoArray[1] );

    printf("L = (matlab base-1), upper triangle is don't care \n");
    printMatrix(m, m, L0, lda, "L0");
    printf("=====\n");

    /*
     * step 4: solve A0*X0 = B0
     *        | 1 |        | 10.5 |
     *   B0 = | 1 |,  X0 = | -2.5 |
     *        | 1 |        | -1.5 |
     */
    status = cusolverDnDpotrsBatched(
            handle,
            uplo,
            m,
            nrhs, /* only support rhs = 1*/
            d_Aarray,
            lda,
            d_Barray,
            ldb,
            d_infoArray,
            batchSize);

    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(infoArray, d_infoArray, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(X0 , Barray[0], sizeof(double)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    cudaDeviceSynchronize();

    printf("info = %d\n", infoArray[0]);
    assert( 0 == infoArray[0] );

    printf("X0 = (matlab base-1)\n");
    printMatrix(m, 1, X0, ldb, "X0");
    printf("=====\n");

    /* free resources */
    if (d_Aarray    ) cudaFree(d_Aarray);
    if (d_Barray    ) cudaFree(d_Barray);
    if (d_infoArray ) cudaFree(d_infoArray);

    if (handle      ) cusolverDnDestroy(handle);
    if (stream      ) cudaStreamDestroy(stream);

    cudaDeviceReset();
}