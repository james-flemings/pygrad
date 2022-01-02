#include "Neuron.h"

int main() {
  std::default_random_engine generator(time(0));
  std::normal_distribution<double> distribution(0.0, 1.0);
  std::string activationFunction = "Sigmoid";
  Neuron *neuron;
  neuron = new Neuron(3, activationFunction, generator, distribution);
  neuron->weights[0] = 3.0;
  neuron->weights[0] = 1.0;
  neuron->weights[0] = 1.0;

  std::vector<double> inputs = {1.0, 1.0};
  double output = neuron->getOutput(inputs);

  std::cout << "The ouput of the neuron is " << output << std::endl;
  return 0;
}