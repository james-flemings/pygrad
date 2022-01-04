#ifndef LAYER_H
#define LAYER_H

#include "Initializer.h"
#include "Neuron.h"
#include <any>

/*
https://stackoverflow.com/questions/7968023/c-virtual-template-method
*/
class Layer {
public:
  Layer();
  Layer(int units, int inputSize = 0,
        const std::string activation = std::string(),
        const std::string initialization = std::string());
  template <typename T> T getOutput(T inputs) {
    std::any res = getOutputImpl(inputs);
    return std::any_cast<T>(res);
  }
  void initializeWeights(Initializer &initializer);
  int totalParameters();
  int inputSize;

protected:
  virtual std::any getOutputImpl(std::any inputs) = 0;
  // virtual std::unique_ptr<Layer> cloneBase() = 0;
  int units;
  std::string activation, initialization;
  std::vector<Neuron> neurons;
};

#endif