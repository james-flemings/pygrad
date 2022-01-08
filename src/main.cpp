#include "../eigen/Eigen/Eigen"
#include "Model.h"
#include <algorithm>
#include <any>
#include <fstream>
#include <iostream>
#include <random>

using namespace Eigen;

// typedef std::vector<std::tuple<VectorXd, VectorXd>> data;

void readIrisData(const std::string &f_name, data &hold_data);

int main() {
  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<DenseLayer>(5, 4, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(3, 0, "Sigmoid"));
  Model model(layers);
  model.summary();

  data hold_data;
  VectorXd x, y;
  readIrisData("Iris.csv", hold_data);
  /*
  for (int i = 0; i < 5; i++) {
    auto [x, y] = hold_data[i];
    std::cout << "X: " << std::endl;
    std::cout << x << std::endl;
  }
  */

  model.train(hold_data, 20, 32, 2.0, 0);

  return 0;
}

void readIrisData(const std::string &f_name, data &hold_data) {
  std::ifstream i_file;
  std::string cell;
  i_file.open(f_name);
  std::vector<double> x, y;
  double max[4] = {0.0, 0.0, 0.0, 0.0};
  double min[4] = {100.0, 100.0, 100.0, 100.0};

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
      if (std::stod(cell) > max[i - 1]) {
        max[i - 1] = std::stod(cell);
      }
      if (std::stod(cell) < min[i - 1]) {
        min[i - 1] = std::stod(cell);
      }
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
  for (auto &data : hold_data) {
    auto [x, y] = data;
    for (int i = 0; i < 4; i++) {
      std::get<0>(data)[i] = (x(i) - min[i]) / (max[i] - min[i]);
    }
  }
}