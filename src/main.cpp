#include "Initializer.h"
#include "Layer.h"
#include "Neuron.h"

int main() {
  std::default_random_engine generator(time(0));
  std::normal_distribution<double> distribution(0.0, 1.0);
  std::string activationFunction = "Sigmoid";
  Neuron *neuron;
  Initializer initializer(time(0));
  auto fn =
      std::bind(&Initializer::randomNormal, initializer, std::placeholders::_1);
  neuron = new Neuron(3, activationFunction, fn);
  neuron->weights[0] = 3.0;
  neuron->weights[0] = 1.0;
  neuron->weights[0] = 1.0;

  std::vector<double> inputs = {1.0, 1.0};
  double output = neuron->getOutput(inputs);

  std::cout << "The ouput of the neuron is " << output << std::endl;
  delete neuron;

  Layer layer(3, 4, "Sigmoid", "Random");
  layer.initializeWeights(initializer);
  std::cout << layer.totalParameters() << std::endl;

  return 0;
}