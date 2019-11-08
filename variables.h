namespace cococat {
class variables{
public:
    variables(int dimension):dimension_(dimension){
        id_ = global_vertex_id++;
    }
    virtual ~variables();
    void setVariables(const Eigen::VectorXd &variables) { variables_ = variables;}
    virtual void Plus(const Eigen::VectorXd &delta){variables_ += delta};
protected:
    int variablesID_;
    int dimension_;
    Eigen::VectorXd variables_;
    static globalvariablesID;
};
}
