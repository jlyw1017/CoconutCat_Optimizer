class optimizer{
private:
    //Actual_Predicted_Reduction_Ratio
    double Damping_Factor;
    double AP_Reduction_Ratio;
    double Trust_Radius;
    double Trust_Radius_global_bound;
    double Trust_bound = 0.25;
    int Max_Steps;
    int current_Step = 0;
    int Damping_Factor_Update_Strategy = 1;// 0 Marquardt 1 for Nielsen
    int LINEAR_SOLVER_TYPE = 0; // 0 for Eigen 1 for CUDA
    double (*target_func)(Eigen::VectorXd);
    //Eigen::MatrixXd A,J;
    Eigen::SparseMatrix<double,Eigen::RowMajor> H,J;
    Eigen::VectorXd x,p,r,g;
    double error_a = 10e-6;
    double error_b = 10e-6;

public:
    optimizer(){

    }

    Eigen::VectorXd linear_solver(
            Eigen::SparseMatrix<double,Eigen::RowMajor> H_,
            double namuda,
            Eigen::VectorXd b){

        Eigen::VectorXd x;
        Eigen::SparseMatrix<double,Eigen::RowMajor> A = H_ + namuda*Eigen::MatrixXd::Identity();

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
/*
    void trust_region(double x,double p){
        bool found;
        while(not found||current_Step>=Max_Steps){
            p = linear_solver();
            AP_Reduction_Ratio = target_func(x)-target_func(x+p)/(model(0)-model(p));
            if(AP_Reduction_Ratio<1/4){
                Trust_Radius = Trust_Radius/4;
            }
            else{
                if(AP_Reduction_Ratio>0.75 && p == Trust_Radius){
                    Trust_Radius = std::min(Trust_Radius,Trust_Radius_global_bound);
                }
                // else Trust_Radius stay the same
            }
            if(AP_Reduction_Ratio>Trust_bound){
                x = x+p;
            }

            ++current_Step;
        }
    }
*/
    void Levenberg_maarquardt(){
        bool found;
        double v = 2;
        double oneThird = 1/3;
        H = J.transpose()*J;
        while(not found||current_Step>=Max_Steps){
            p = linear_solver(H,Damping_Factor,g);
            found = stop_condition();
            if(found){
                found = (p.norm() < error_b*(target_func(x)-error_b));
            }
            else{
                x += p;
                AP_Reduction_Ratio = 2*(target_func(x)-target_func(x+p))/(p.transpose()*(Damping_Factor*p+g));
                if(AP_Reduction_Ratio>0){
                    H = J.transpose()*J;
                    g = -J.transpose()*r;
                    found = ((g.lpNorm<Eigen::Infinity>())<= error_a);
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

    void run_optimize(){
        double p,x;
    }
};