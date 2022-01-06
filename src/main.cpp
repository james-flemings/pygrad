#include "../eigen/Eigen/Eigen"
//#include "Model.h"
#include <any>
#include <iostream>

using namespace Eigen;

int main() {
  VectorXd v{{-1.0, 2.0}};
  MatrixXd m = v * v.transpose();
  // VectorXd u = v * v.transpose();

  std::cout << "Matrix:" << std::endl;
  std::cout << m << std::endl;

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