namespace cococat {
class factor{
public:
    factor(){

    }
    virtual ~factor();
    bool addfactor(){

    }

protected:
    int factorID; //
    std::vector<std::shared_ptr<Vertex>> vecVariables_; // 该边对应的顶点
    Eigen::vectorXd residual_;                 // 残差
    std::vector<MatXX> jacobians_;  // 雅可比，每个雅可比维度是 residual x vertex[i]
    MatXX information_;             // 信息矩阵
    VecX observation_;              // 观测信息
};






}
