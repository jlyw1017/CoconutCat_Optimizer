#include <iostream>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
//#include "linearSolver.h"
#include <opencv2/opencv.hpp>
#include "linearSolver_SparseCholesky.h"
//#include "Optimizer_Leveberg_Marquardt.h"

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double

void showMatrix(Eigen::SparseMatrix<double,Eigen::RowMajor> S){
    std::cout << "S:" << std::endl;
    std::cout << "InnerIndexPtr:" << std::endl;
    for(int i=0;i<S.nonZeros();i++){
        std::cout  << S.innerIndexPtr()[i] << " ";
    }
    std::cout << std::endl << "S outerIndexPtr:" << std::endl;
    for(int i=0;i<S.outerSize();i++) {

        std::cout  << S.outerIndexPtr()[i] << " ";
    }
    std::cout << std::endl << "S valuePtr:" << std::endl;
    for(int i=0;i<S.nonZeros();i++) {
        std::cout << S.valuePtr()[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

// A Example for Linear Solver
void Linear_Solver_test(){
    // Prepare data
    std::cout << "Data:" << std::endl;
    Eigen::MatrixXd A(5,5);
    A << 1,    0,   0,   0,   0,
            0,    2,   0,   0,   0,
            0 ,   0.1,   3,   0,   0,
            0 ,   0,   0,   4,   0,
            0.1,    0.1,   0.1,   0,   1;

    Eigen::SparseMatrix<double,Eigen::RowMajor> S = A.sparseView(1,0.001);
    std::cout << "A:" << std::endl << A << std::endl;
    showMatrix(S);
    Eigen::VectorXd b(5);
    b << 1,1,1,1,1;

    // Eigen Solver
    std::cout << "Eigen Solver Begin!" << std::endl;
    Eigen::SimplicialCholesky<SpMat> chol(S);  // performs a Cholesky factorization of A
    Eigen::VectorXd x_eigen = chol.solve(b);         // use the factorization to solve for the given right hand side
    std::cout << "Result:" << std::endl << x_eigen << std::endl << std::endl;

    // Cuda Solver
    std::cout << "Cuda Solver Begin!" << std::endl;
    Eigen::VectorXd x_cuSolver = cuda_Solver(S,b);
    std::cout << "Result:" << std::endl << x_cuSolver << std::endl << std::endl;
}

int main() {
    std::cout << "CoconutCat is CUTE!" << std::endl;
    Linear_Solver_test();
    std::cout << "CoconutCat Optimization Finished!" << std::endl;
    return 0;
}

