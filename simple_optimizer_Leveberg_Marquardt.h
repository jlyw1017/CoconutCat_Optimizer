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
    void run_optimize(){
        std::cout << "Optimize Run"  << std::endl;
        bool OpCondition = Levenberg_maarquardt();
        std::cout << "Optimize Process Finised? " << std::endl << OpCondition << std::endl;
        std::cout << "Result:" << std::endl;
        for(int i =0; i < variables_.size();i++){
            std::cout << variables_[i] << std::endl;
        }
    }

    bool Levenberg_maarquardt(){
        bool foundFLAG;
        double v = 2;
        double oneThird = 1/3;
        updateJacobian();
        updateResidual();
        H_ = J_.transpose()*J_;
        //std::cout << "J_:"  << std::endl << J_;
        //std::cout << "H_:"  << std::endl << H_;

        /*
         * // Dense version
        HD_ = JD_.transpose()*JD_;
        std::cout << "JD_:"  << std::endl << JD_;
        std::cout << "HD_:"  << std::endl << HD_;
*/
        Damping_Factor = Damping_Factor_InitialFactor*initialLambda(H_);
        //std::cout << "Damping_Factor:"  << std::endl << Damping_Factor << std::endl;
        g = -J_.transpose()*r;
        //std::cout << "g:"  << std::endl << g;
        while(not foundFLAG && current_Step < Max_Steps){
            std::cout << std::endl << "Iteration:" << current_Step << std::endl;
            p = linear_solver(H_,Damping_Factor,g);
            std::cout << "x:" << variables_ <<std::endl;
            //std::cout << "P:" << p <<std::endl;

            foundFLAG = stop_condition();
            if(foundFLAG){
                foundFLAG = (p.norm() < error_b*((target_func_)(variables_,input_TF,observations)-error_b));
            }
            else{
                AP_Reduction_Ratio = 2*((*target_func_)(variables_,input_TF,observations)-(*target_func_)(variables_+p,input_TF,observations))
                        /(p.transpose()*(Damping_Factor*p+g));
                variables_ += p;
                if(AP_Reduction_Ratio>0){
                    updateJacobian();
                    updateResidual();
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
        return 1;
    }

    Eigen::VectorXd linear_solver(
            Eigen::SparseMatrix<double,Eigen::RowMajor> H_,
            double namuda,
            Eigen::VectorXd b){
        //std::cout << "Line Solver Begin"   << std::endl;
        int size = H_.rows();
        Eigen::VectorXd x(b.size());
        Eigen::SparseMatrix<double,Eigen::RowMajor> A = H_ + namuda*Eigen::MatrixXd::Identity(size,size);

        if(LINEAR_SOLVER_TYPE == 0){
            //std::cout << "Eigen Solve"   << std::endl;
            Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(A);  // performs a Cholesky factorization of A
            //std::cout << "Cholesky Finished" << chol.solve(b) << std::endl;
            x = chol.solve(b);
        }
        else if(LINEAR_SOLVER_TYPE == 1){
            //std::cout << "Cuda Solve"   << std::endl;
            x = cuda_Solver(A,b);
        }

        return x;
    }

    double initialLambda(Eigen::SparseMatrix<double,Eigen::RowMajor> H_){
        int size = H_.rows();
        double max = H_.coeff(0,0);
        for(int i=1;i<size;++i){
            max = max < H_.coeff(i,i) ? H_.coeff(i,i):max;
        }
        return max;
    }

    bool stop_condition(){
        return false;
    }
};
