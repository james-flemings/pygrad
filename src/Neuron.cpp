#include "Neuron.h"

Neuron::Neuron(int size, const std::string &activation,
               std::function<void(std::vector<double>)> initializer) {
  /*
  Initilize Neuron by determining activation funciton and intializing the
  weights
  */
  this->activation = activation;
  this->size = size + 1;
  this->weights.resize(this->size);
  initializer(this->weights);
}

double Neuron::getOutput(std::vector<double> inputs) {
  /*
  Calculate output of a neuron given the inputs with corresponding weights (and
  bias)
  */
  double output = 0.0;
  if (!this->activation.compare("Sigmoid")) {
    double product =
        std::inner_product(this->weights.begin() + 1, this->weights.end(),
                           inputs.begin(), this->weights[0]);
    output = 1 / (1 + exp(-product));
  } else {
    throw std::domain_error(
        "Invalid activation function in get output. This should not occur.");
  }
  return output;
}
