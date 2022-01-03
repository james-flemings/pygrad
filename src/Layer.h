#ifndef LAYER_H
#define LAYER_H

#include "Initializer.h"
#include "Neuron.h"

class Layer {
public:
  Layer(int units, int inputSize = 0,
        const std::string activation = std::string(),
        const std::string initialization = std::string());
  virtual ~Layer(){};
  // virtual std::vector<double> getOutput(std::vector<double> inputs);
  void initializeWeights(Initializer &initializer);
  int totalParameters();

protected:
  int units, inputSize;
  std::string activation, initialization;
  std::vector<Neuron> neurons;
};

#endif