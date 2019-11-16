namespace cococat {
class residual;
class problem{
public:
    problem(){

    }
    virtual ~problem();

    bool addResidual(shared_ptr<residual> residuals) {
        if (edges_.find(edge->Id()) == edges_.end()) {
            edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
        } else {
            // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
            return false;
        }

        for (auto &vertex: edge->Verticies()) {
            vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
        }
        return true;
    }


protected:
    



};






}
