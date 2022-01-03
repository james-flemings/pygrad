#include "DenseLayer.h"

DenseLayer::DenseLayer(int units, int inputSize, const std::string activation,
                       const std::string initialization)
    : Layer(units, inputSize, activation, initialization) {}

std::any DenseLayer::getOutputImpl(std::any inputs) {
  std::vector<double> output;
  for (auto &n : this->neurons)
    output.push_back(n.getOutput(std::any_cast<std::vector<double>>(inputs)));
  return output;
}