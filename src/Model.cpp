#include "Model.h"

Model::Model(const std::vector<std::unique_ptr<Layer>> &layers) {
  // First layer must contain the input size
  if (layers[0]->inputSize == 0)
    throw std::invalid_argument("First layer must contain input size");

  Initializer initializer(time(0));
  int prevInputSize = 0;
  for (auto &layer : this->layers) {
    layer->inputSize = prevInputSize ? layer->inputSize == 0 : layer->inputSize;
    prevInputSize = layer->inputSize;
    layer->initializeWeights(initializer);
    this->layers.push_back(std::move(layer));
  }
}

int Model::totalParameters() {
  int total = 0;
  for (auto &layer : this->layers)
    total += layer->totalParameters();
  return total;
}