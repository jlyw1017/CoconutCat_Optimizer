#include <random>
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double

void showMatrix(Eigen::SparseMatrix<double, Eigen::RowMajor> S) {
    std::cout << "S:" << std::endl;
    std::cout << "InnerIndexPtr:" << std::endl;
    for (int i = 0; i < S.nonZeros(); i++) {
        std::cout << S.innerIndexPtr()[i] << " ";
    }
    std::cout << std::endl << "S outerIndexPtr:" << std::endl;
    for (int i = 0; i < S.outerSize(); i++) {

        std::cout << S.outerIndexPtr()[i] << " ";
    }
    std::cout << std::endl << "S valuePtr:" << std::endl;
    for (int i = 0; i < S.nonZeros(); i++) {
        std::cout << S.valuePtr()[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

// A Example for Linear Solver
void Linear_Solver_test() {
    // Prepare data
    std::cout << "Data:" << std::endl;
    Eigen::MatrixXd A(5, 5);
    A << 1, 0, 0, 0, 0,
            0, 2, 0, 0, 0,
            0, 0.1, 3, 0, 0,
            0, 0, 0, 4, 0,
            0.1, 0.1, 0.1, 0, 1;

    Eigen::SparseMatrix<double, Eigen::RowMajor> S = A.sparseView(1, 0.001);
    std::cout << "A:" << std::endl << A << std::endl;
    showMatrix(S);
    Eigen::VectorXd b(5);
    b << 1, 1, 1, 1, 1;

    // Eigen Solver
    std::cout << "Eigen Solver Begin!" << std::endl;
    Eigen::SimplicialCholesky<SpMat> chol(S);  // performs a Cholesky factorization of A
    Eigen::VectorXd x_eigen = chol.solve(b);         // use the factorization to solve for the given right hand side
    std::cout << "Result:" << std::endl << x_eigen << std::endl << std::endl;

    // Cuda Solver
    std::cout << "Cuda Solver Begin!" << std::endl;
    Eigen::VectorXd x_cuSolver = cuda_Solver(S, b);
    std::cout << "Result:" << std::endl << x_cuSolver << std::endl << std::endl;
}

class Curvefitting:public simpleoptimizer{
public:
    Curvefitting(){};
    ~Curvefitting();

    void addFactor(Eigen::VectorXd input_TargetFunction,Eigen::VectorXd z) {
        input_TF.conservativeResize(input_TF.size() + input_TargetFunction.size());
        input_TF.tail(input_TargetFunction.size()) = input_TargetFunction;
        //std::cout << "Input added!" << std::endl;
        //std::cout << "input_TF:" << input_TF << std::endl;

        observations.conservativeResize(observations.size() + z.size());
        observations.tail(z.size()) = z;
        //std::cout << "Observation added!" << std::endl;
        //std::cout << "Observations:" << observations << std::endl;

        r.resize(z.size());
        std::vector<double> rtemp;
        for(int i=0;i<z.size();i++){
            double error = computeResidual(input_TargetFunction[i],z[i]);
            rtemp.push_back(error);
        }
        Eigen::VectorXd rtempEigen = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(rtemp.data(), rtemp.size());
        r = rtempEigen;
    }

    void initialize(Eigen::VectorXd variablesInit){
        variables_ = variablesInit;}

    virtual double computeResidual(double x,double y) override {
        double r;
        r = std::exp(variables_(0)*x*x + variables_(1)*x + variables_(2)) - y; //)
        return r;// 构建残差
    }

    virtual void updateResidual() override {
        r.resize(observations.size());
        std::vector<double> rtemp;
        for(int i=0;i<observations.size();i++){
            double error = computeResidual(input_TF[i],observations[i]);
            rtemp.push_back(error);
        }
        Eigen::VectorXd rtempEigen = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(rtemp.data(), rtemp.size());
        r = rtempEigen;
    }

    virtual Eigen::MatrixXd computeJacobian(double x) override {
        double exp_y = std::exp(variables_(0)*x*x + variables_(1)*x + variables_(2));
        Eigen::Matrix<double, 1, 3> jaco_abc;  // 误差为1维，状态量 3 个，所以是 1x3 的雅克比矩阵
        jaco_abc << exp_y*x * x , exp_y*x  , exp_y;
        //std::cout << "x" << x << "jaco_abc" << jaco_abc << std::endl;
        return jaco_abc;
    }

    virtual void updateJacobian() override{
        std::vector<double> vecJ;
        for(int i=0;i<observations.size();i++){
            Eigen::MatrixXd temp = computeJacobian(input_TF[i]);
            vecJ.push_back(temp(0,0));
            vecJ.push_back(temp(0,1));
            vecJ.push_back(temp(0,2));
        }
        Eigen::MatrixXd Jtemp;
        Jtemp = Eigen::MatrixXd::Map(&vecJ[0], 3, observations.size());
        //std::cout << "J_:"  << std::endl << J_;
        JD_ = Jtemp.transpose();
        J_ = Jtemp.transpose().sparseView(1, 0.0001);
    };

    virtual void setTargetFunction(double (*target_func)(Eigen::VectorXd,Eigen::VectorXd,Eigen::VectorXd)) override {
        target_func_ = target_func;
    }
};

double expCurve(Eigen::VectorXd parameters,Eigen::VectorXd x,Eigen::VectorXd z){
    double sum=0;
    for(int i=0;i<x.size();i++){
        sum += std::pow(std::exp(parameters(0)*x(i)*x(i) + parameters(1)*x(i) + parameters(2))-z[i],2);
    }
    return sum/2;
}

void Simple_Optimizer_test(){
    std::cout << "Optimizer Begein!" << std::endl;
    double a=1.0, b=2.0, c=1.0;         // line fitting for exp( a*x*x + b*x + c )  我们以这个模型作为测试对象
    int N = 1000;                       // Number of Datapoints   数据点数量
    double w_sigma= 1;                 // Noise Sigma  高斯分布方差

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.,w_sigma);

    Eigen::VectorXd variablesInit(3);
    variablesInit << 2,4,2;

    std::vector<double> sin;
    std::vector<double> sz;
    for(int i = 0; i < N; ++i) {
        double x = i/1000.;
        double n = noise(generator);
        // Oberservation 观测 y
        //double y = a*x*x + b*x + c  + n;
        double y = std::exp(a*x*x + b*x + c) + n; //
        sin.push_back(x);
        sz.push_back(y);
    }
    Eigen::VectorXd inputx = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(sin.data(), sin.size());
    Eigen::VectorXd z = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(sz.data(), sz.size());
    //std::cout << "inputx:" << std::endl << inputx << std::endl;
    //std::cout << "z:" << std::endl << z << std::endl;

    Curvefitting* CurveOp = new Curvefitting();
    CurveOp->initialize(variablesInit);
    CurveOp->setTargetFunction(&expCurve);
    //std::cout << "Optimizer Initialized!" << std::endl;
    CurveOp->addFactor(inputx,z);
    /// 使用 LM 求解
    CurveOp->run_optimize();

    std::cout << "-------After optimization, we got these parameters :" << std::endl;
    std::cout << "-------ground truth: " << std::endl;
    std::cout << "1.0,  2.0,  1.0" << std::endl;
}

