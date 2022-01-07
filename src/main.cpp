#include "../eigen/Eigen/Eigen"
#include "Model.h"
#include <algorithm>
#include <any>
#include <fstream>
#include <iostream>
#include <random>

using namespace Eigen;

void readIrisData(const std::string &f_name, data &hold_data);

int main() {
  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<DenseLayer>(3, 4, "Sigmoid"));
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

  data hold_data;
  VectorXd x, y;
  readIrisData("Iris.csv", hold_data);

  model.train(hold_data, 5, 32, 0.1, 0);

  return 0;
}

void readIrisData(const std::string &f_name, data &hold_data) {
  std::ifstream i_file;
  std::string cell;
  i_file.open(f_name);
  std::vector<double> x;

  std::map<std::string, int> vectorize;
  vectorize.insert(std::pair<std::string, int>("Iris-setosa", 0));
  vectorize.insert(std::pair<std::string, int>("Iris-versicolor", 1));
  vectorize.insert(std::pair<std::string, int>("Iris-virginica", 2));

  int i = 0;
  std::getline(i_file, cell);
  while (std::getline(i_file, cell, ',')) {
    // long, complicated if statement checking if cell is a digit
    if (i == 0) {
      i += 1;
      continue;
    } else {
      x.push_back(std::stod(cell));
      i += 1;
    }
    if (i == 5) {
      VectorXd label;
      label.setZero(3);
      std::getline(i_file, cell);
      label(vectorize[cell]) = 1.0;
      hold_data.push_back(
          std::make_tuple(Map<VectorXd>(x.data(), x.size()), label));
      x = std::vector<double>();
      i = 0;
    }
  }
}