#include "../eigen/Eigen/Eigen"
#include "Model.h"
#include <any>
#include <iostream>

using namespace Eigen;

int main() {
  /*
  VectorXd v1{{1.0, 2.0}};
  std::cout << "The vector is \n" << v1 << std::endl;

  for (auto &v : v1) {
    v = 4;
  }
  std::cout << "The vector is now \n" << v1 << std::endl;

  std::cout << "The size is \n" << v1.size() << std::endl;
  */

  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<DenseLayer>(20, 100, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(10, 0, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(5, 0, "Sigmoid"));
  Model model(layers);
  model.summary();

  return 0;
}