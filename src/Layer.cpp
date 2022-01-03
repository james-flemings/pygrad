#include "Layer.h"

Layer::Layer(int units, int inputSize = 0,
             const std::string activation = std::string(),
             const std::string initializer = std::string()) {
  /*
  Layer class is the general base class for specifc layers
  Default value for activation is Sigmoid.

  If user requests unimplemented activation function or initializer, throw
  exception.

  Right now weights (and bias) are initialized by randomly selecting a value
  from a normal distribution (Random)
  */
  if (activation.empty())
    this->activation = "Sigmoid";
  else if (initializer.compare("Sigmoid"))
    this->activation = activation;
  else
    throw std::invalid_argument("Invalid activation function");

  if (initializer.empty())
    this->initializer = "Random";
  else if (initializer.compare("Random"))
    this->initializer = initializer;
  else
    throw std::invalid_argument("Invalid initializer scheme");

  this->units = units;
  this->inputSize = inputSize;
}

double Layer::initializeWeights(Initializer &initializer, int seed) {
  for (int i = 0; i < this->units; i++)
    this->neurons.push_back(
        Neuron(this->inputSize, this->activation, initializer.randomNormal));
}
