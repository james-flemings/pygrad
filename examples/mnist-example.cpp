#include "../eigen/Eigen/Eigen"
#include "../mnist/include/mnist/mnist_reader.hpp"
#include "Model.h"

#include <iostream>
#include <string>

std::string MNIST_DATA_LOCATION = "../mnist";

using namespace Eigen;

typedef std::vector<std::tuple<VectorXd, VectorXd>> data;

void readMnist(data &train_data, data &valid_data);

int main(int argc, char *argv[]) {
  data train_data, valid_data;
  readMnist(train_data, valid_data);

  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<DenseLayer>(30, 784, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(10, 0, "Sigmoid"));
  Model model(layers);
  model.summary();

  model.train(train_data, 30, 10, 2.0, 0, valid_data);

  return 0;
}

void readMnist(data &train_data, data &valid_data) {
  mnist::MNIST_dataset<std::vector, std::vector<double>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, double, uint8_t>(
          MNIST_DATA_LOCATION);
  for (int i = 0; i < dataset.training_images.size(); i++) {
    VectorXd label;
    label.setZero(10);
    label(dataset.training_labels[i]) = 1.0;
    Map<VectorXd> data(dataset.training_images[i].data(),
                       dataset.training_images[i].size());
    data = data.array() * (double)1 / 255;
    train_data.push_back(std::make_tuple(data, label));
  }
  for (int i = 0; i < dataset.test_images.size(); i++) {
    VectorXd label;
    label.setZero(10);
    label(dataset.test_labels[i]) = 1.0;
    Map<VectorXd> data(dataset.test_images[i].data(),
                       dataset.test_images[i].size());
    data = data.array() * (double)1 / 255;
    valid_data.push_back(std::make_tuple(data, label));
  }
}