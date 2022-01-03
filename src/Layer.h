#ifndef LAYER_H
#define LAYER_H

#include "Initializer.h"
#include "Neuron.h"

class Layer {
public:
  Layer(int units, int inputSize = 0,
        const std::string activation = std::string(),
        const std::string initializer = std::string());
  virtual std::vector<double> getOutput(std::vector<double> inputs);
  double initializeWeights(Initializer &initializer, int seed);

protected:
  int units, inputSize;
  std::string activation, initializer;
  std::vector<Neuron> neurons;
};

#endif