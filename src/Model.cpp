#include "Model.h"

Model::Model(std::vector<std::unique_ptr<Layer>> &layers) {
  // First layer must contain the input size
  if (layers[0]->inputSize == 0)
    throw std::invalid_argument("First layer must contain input size");

  Initializer initializer(time(0));
  int prevInputSize = 0;

  this->layers.resize(layers.size());
  std::move(layers.begin(), layers.end(), this->layers.begin());

  for (auto &layer : this->layers) {
    layer->inputSize = layer->inputSize == 0 ? prevInputSize : layer->inputSize;
    prevInputSize = layer->inputSize;
    layer->initializeWeights(initializer);
  }
}

int Model::totalParameters() {
  int total = 0;
  for (auto &layer : this->layers)
    total += layer->totalParameters();
  return total;
}

void Model::printModel() {
  std::string row = "---------------------------";
  std::cout << "\nLayer \t Units \t Parameters \n";
  std::cout << row << std::endl;
  std::cout << "Input \t " << layers[0]->inputSize << " \t "
            << "__ \n";
  for (auto &layer : this->layers) {
    std::cout << row << std::endl;
    std::cout << "Dense \t " << layer->getUnits() << " \t "
              << layer->totalParameters() << std::endl;
    ;
  }
  std::cout << row << std::endl;
  std::cout << "Total Prameters: " << this->totalParameters() << "\n\n";
}