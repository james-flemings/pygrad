#include "../eigen/Eigen/Eigen"
//#include "Model.h"
#include <any>
#include <iostream>

using namespace Eigen;

int main() {
  VectorXd v{{1.0, 2.0}};
  VectorXd u = v.array() * v.array();
  MatrixXd m(2, 2);

  m.row(0) = v;
  m.row(1) = v;

  std::cout << v * v.transpose() << std::endl;
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