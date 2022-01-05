#include "DenseLayer.h"

DenseLayer::DenseLayer(int units, int inputSize, const std::string activation,
                       const std::string initialization)
    : Layer(units, inputSize, activation, initialization) {}

std::any DenseLayer::getOutputImpl(std::any inputs) {
  VectorXd output;
  output.resize(this->neurons.size());
  for (int i = 0; i < this->neurons.size(); i++) {
    output(i) = this->neurons[i].getOutput(std::any_cast<VectorXd>(inputs));
  }
  return output;
}