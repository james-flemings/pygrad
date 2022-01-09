#include "../eigen/Eigen/Eigen"
#include "../mnist/include/mnist/mnist_reader.hpp"
#include "../mnist/include/mnist/mnist_utils.hpp"
#include "Model.h"

#include <iostream>
#include <string>

std::string MNIST_DATA_LOCATION = "../mnist";

using namespace Eigen;

void readMnist(data &train_data, data &valid_data);

int main(int argc, char *argv[]) {
  data train_data, valid_data;
  VectorXd x, y;
  readMnist(train_data, valid_data);
  auto [x, y] = train_data[0];
  std::cout << "X: " << std::endl;
  std::cout << x << std::endl;
  std::cout << "Y: " << std::endl;
  std::cout << y << std::endl;
  return 0;
}

void readMnist(data &train_data, data &valid_data) {
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
          MNIST_DATA_LOCATION);
  for (int i = 0; i < dataset.training_images.size(); i++) {
    VectorXd label(10);
    label(dataset.training_labels[i]) = 1.0;
    VectorXd data(dataset.training_images[i].data());
    data = data.array() * (double)1 / 255;
    train_data.push_back(std::make_tuple(data, label));
  }
  for (int i = 0; i < dataset.test_images.size(); i++) {
    VectorXd label(10);
    label(dataset.test_labels[i]) = 1.0;
    VectorXd data(dataset.test_images[i].data());
    data = data.array() * (double)1 / 255;
    valid_data.push_back(std::make_tuple(data, label));
  }
}