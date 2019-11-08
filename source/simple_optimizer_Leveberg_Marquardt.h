class simpleoptimizer{
protected:
    // Commen Parameters
    int Max_Steps;
    int Damping_Factor_Update_Strategy = 1;// 0 Marquardt 1 for Nielsen
    int LINEAR_SOLVER_TYPE = 0; // 0 for Eigen 1 for CUDA

    double error_a = 10e-6;
    double error_b = 10e-6;
    double (*target_func_)(Eigen::VectorXd);
    //Eigen::MatrixXd A,J;

    // LM Parameters
    double Damping_Factor;
    double AP_Reduction_Ratio;     //Actual_Predicted_Reduction_Ratio

    // Trust Region Parameters
    double Trust_Radius;
    double Trust_Radius_global_bound;
    double Trust_bound = 0.25;

    // Variables
    int current_Step = 0;
    Eigen::SparseMatrix<double,Eigen::RowMajor> H_,J_;
    Eigen::VectorXd variables_,p,r,g;  // x_ is your optimization target variables
    Eigen::VectorXd input_TF;  // Input of your Target Function
    Eigen::VectorXd observations;
public:
    simpleoptimizer();
    ~simpleoptimizer();
    // Functions you need to override for a particular problem

    virtual void updateJacobian();
    virtual Eigen::MatrixXd computeJacobian(double x);
    virtual void updateResidual();
    virtual double computeResidual(double x,double y);

    void run_optimize() {
        std::cout << "Optimize Run"  << std::endl;
        bool OpCondition = Levenberg_maarquardt();
        std::cout << "Optimize Process Finised?" << OpCondition << std::endl;
        std::cout << "Result:" << std::endl;
        for(int i =0; i < variables_.size()-1;i++){
            std::cout << variables_[i] << std::endl;
        }
    }

    bool Levenberg_maarquardt(){
        bool foundFLAG;
        double v = 2;
        double oneThird = 1/3;
        std::cout << "Calculate H"  << std::endl;
        H_ = J_.transpose()*J_;
        std::cout << "H:"  << std::endl << H;
        while(not foundFLAG||current_Step>=Max_Steps){
            p = linear_solver(H_,Damping_Factor,g);
            foundFLAG = stop_condition();
            if(foundFLAG){
                foundFLAG = (p.norm() < error_b*((target_func_)(variables_)-error_b));
            }
            else{
                variables_ += p;
                AP_Reduction_Ratio = 2*((*target_func_)(variables_)-(*target_func_)(variables_+p))/(p.transpose()*(Damping_Factor*p+g));
                if(AP_Reduction_Ratio>0){
                    updateJacobian();
                    H_ = J_.transpose()*J_;
                    g = -J_.transpose()*r;
                    foundFLAG = ((g.lpNorm<Eigen::Infinity>())<= error_a);
                    Damping_Factor = Damping_Factor*std::max(oneThird,1-std::pow(2*AP_Reduction_Ratio-1,3));
                    v = 2;
                }
                else{
                    Damping_Factor = Damping_Factor*v;
                    v = 2*v;
                }
                ++current_Step;
            }
        }
    }

    Eigen::VectorXd linear_solver(
            Eigen::SparseMatrix<double,Eigen::RowMajor> H_,
            double namuda,
            Eigen::VectorXd b){
        int size = b.size();
        Eigen::VectorXd x(b.size());
        Eigen::SparseMatrix<double,Eigen::RowMajor> A = H_ + namuda*Eigen::MatrixXd::Identity(size,size);

        if(LINEAR_SOLVER_TYPE == 0){
            Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(A);  // performs a Cholesky factorization of A
            x = chol.solve(b);
        }
        else if(LINEAR_SOLVER_TYPE == 1){
            cuda_Solver(A,b);
        }

        return x;
    }

    bool stop_condition(){
        return false;
    }
};
