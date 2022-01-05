#include "../eigen/Eigen/Eigen"
#include <iostream>
//#include "Model.h"

using namespace Eigen;

int main() {
  VectorXd v1;
  v1.resize(2);
  v1(0) = 1.0;
  v1(1) = 2.0;
  VectorXd v2;
  v2.resize(2);
  v2(0) = 1.0;
  v2(1) = 1.0;

  double product = v1.dot(v2);
  std::cout << "The dot product is " << product << std::endl;
  /*
  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<DenseLayer>(20, 100, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(10, 0, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(5, 0, "Sigmoid"));
  Model model(layers);
  model.summary();
  */

  return 0;
}