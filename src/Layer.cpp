#include "Layer.h"

Layer::Layer(int units, int inputSize, const std::string activation,
             const std::string initialization) {
  /*
  Layer class is the general base class for specifc layers
  Default activation function is None.

  If user requests unimplemented activation function or initializer, throw
  exception.

  Right now weights (and bias) are initialized by randomly selecting a value
  from a normal distribution (Random)
  */
  if (activation.empty())
    this->activation = "None";
  else if (!activation.compare("Sigmoid"))
    this->activation = activation;
  else
    throw std::invalid_argument("Invalid activation function");

  if (initialization.empty())
    this->initialization = "Random";
  else if (!initialization.compare("Random"))
    this->initialization = initialization;
  else
    throw std::invalid_argument("Invalid initialization scheme");

  if (units <= 0)
    throw std::invalid_argument("Units for Layers must be positive");

  this->units = units;
  this->inputSize = inputSize;
}

void Layer::initializeWeights(Initializer &initializer) {
  auto fn = [&] {
    if (!this->initialization.compare("Random"))
      return std::bind(&Initializer::randomNormal, initializer,
                       std::placeholders::_1, std::placeholders::_2);
  }();

  for (int i = 0; i < this->units; i++)
    this->neurons.push_back(Neuron(this->inputSize, this->activation, fn));
}

int Layer::totalParameters() {
  int total = 0;
  for (auto &n : this->neurons) {
    total += (n.weights.size() + 1);
  }
  return total;
}

int Layer::getUnits() { return this->units; }

std::string Layer::getActivation() { return this->activation; }

std::string Layer::getInitialization() { return this->initialization; }