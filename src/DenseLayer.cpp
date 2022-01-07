#include "DenseLayer.h"

DenseLayer::DenseLayer(int units, int inputSize, const std::string activation,
                       const std::string initialization)
    : Layer(units, inputSize, activation, initialization) {}

std::any DenseLayer::getOutputImpl(const std::any &inputs) {
  VectorXd z =
      this->getWeights() * std::any_cast<VectorXd>(inputs) + this->getBias();
  return this->sigmoid(z);
}