#include "Neuron.h"

Neuron::Neuron(int size, const std::string &activation,
               std::function<void(VectorXd &, double &)> initializer) {
  /*
  Initilize Neuron by determining activation funciton and intializing the
  weights
  */
  this->activation = activation;
  this->size = size;
  this->weights.resize(this->size);
  initializer(this->weights, this->bias);
}

double Neuron::getOutput(const VectorXd &inputs) const {
  /*
  Calculate output of a neuron given the inputs with corresponding weights (and
  bias)
  */
  double output = 0.0;
  if (!this->activation.compare("Sigmoid")) {
    output = 1 / (1 + exp(-(this->weights.dot(inputs) + this->bias)));
  } else if (!this->activation.compare("None")) {
    output = this->weights.dot(inputs) + this->bias;
  } else {
    throw std::domain_error(
        "Invalid activation function in get output. This should not occur.");
  }
  return output;
}