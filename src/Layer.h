#ifndef LAYER_H
#define LAYER_H

#include "Initializer.h"
#include "Neuron.h"

/*
Using the curiously recurring template pattern (CRTP) taken from:
https://stackoverflow.com/questions/2354210/can-a-class-member-function-template-be-virtual
https://stackoverflow.com/questions/4173254/what-is-the-curiously-recurring-template-pattern-crtp
*/
template <typename T> class Layer {
public:
  Layer(int units, int inputSize = 0,
        const std::string activation = std::string(),
        const std::string initialization = std::string());
  virtual ~Layer(){};
  template <typename inputType> inputType getOutput(inputType inputs) {
    return ((T *)this)->getOutput<inputType>(inputs);
  }
  void initializeWeights(Initializer &initializer);
  int totalParameters();

protected:
  int units, inputSize;
  std::string activation, initialization;
  std::vector<Neuron> neurons;
};

#endif