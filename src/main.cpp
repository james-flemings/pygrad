#include "../eigen/Eigen/Eigen"
#include "Model.h"
#include <any>
#include <iostream>

using namespace Eigen;

int main() {
  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<DenseLayer>(5, 3, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(2, 0, "Sigmoid"));
  Model model(layers);
  model.summary();

  VectorXd input{{1.0, 1.0, 1.0}};
  std::vector<VectorXd> activations;
  model.forwardPass(activations, input);
  int i = 0;
  for (auto &a : activations) {
    std::cout << "Output " << i << std::endl;
    std::cout << a << std::endl;
  }

  std::cout << activations.size() << std::endl;

  return 0;
}