#include <iostream>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

#ifndef COCONUTCAT_CUDA_SOLVER
#define COCONUTCAT_CUDA_SOLVER
#include "linearSolver_SparseCholesky.h"
#endif

class simpleoptimizer{
protected:
    // Commen Parameters
    int Max_Steps = 100;
    int Damping_Factor_Update_Strategy = 1;// 0 Marquardt 1 for Nielsen
    int LINEAR_SOLVER_TYPE = 0; // 0 for Eigen 1 for CUDA

    double error_a = 10e-6;
    double error_b = 10e-6;
    double (*target_func_)(Eigen::VectorXd,Eigen::VectorXd,Eigen::VectorXd);
    //Eigen::MatrixXd A,J;

    // LM Parameters
    double Damping_Factor = 0;
    double Damping_Factor_InitialFactor = 1e-5;
    double AP_Reduction_Ratio;     //Actual_Predicted_Reduction_Ratio

    // Trust Region Parameters
    double Trust_Radius;
    double Trust_Radius_global_bound;
    double Trust_bound = 0.25;

    // Variables
    int current_Step = 0;
    Eigen::MatrixXd JD_, HD_;
    Eigen::SparseMatrix<double,Eigen::RowMajor> H_,J_;
    //Eigen::SparseMatrix<double,Eigen::RowMajor> Information;
    Eigen::VectorXd variables_,p,r,g;  // x_ is your optimization target variables
    Eigen::VectorXd input_TF;  // Input of your Target Function
    Eigen::VectorXd observations;
public:
    simpleoptimizer(){};
    ~simpleoptimizer();
    // Functions you need to override for a particular problem

    virtual void updateJacobian() = 0;
    virtual Eigen::MatrixXd computeJacobian(double x) = 0;
    virtual void updateResidual() = 0;
    virtual double computeResidual(double x,double y) = 0;
    virtual void setTargetFunction(double (*target_func_)(Eigen::VectorXd,Eigen::VectorXd,Eigen::VectorXd)) = 0;
    void run_optimize();
    bool Levenberg_maarquardt();
    Eigen::VectorXd linear_solver(
            Eigen::SparseMatrix<double,Eigen::RowMajor> H_,
            double namuda,
            Eigen::VectorXd b);

    double initialLambda(Eigen::SparseMatrix<double,Eigen::RowMajor> H_);
    bool stop_condition();
};
