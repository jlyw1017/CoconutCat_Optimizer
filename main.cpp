#include <iostream>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

#include "settings.h"
#include "utilis.h"

int main() {
    std::cout << "CoconutCat is CUTE!" << std::endl;
    Linear_Solver_test();
    std::cout << "Line Solver Finished!" << std::endl;
    Simple_Optimizer_test();
    std::cout << "Optimizer Finished!" << std::endl;
    std::cout << "CoconutCat Optimization Finished!" << std::endl;
    return 0;
}


