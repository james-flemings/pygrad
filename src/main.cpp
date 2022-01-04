#include "DenseLayer.h"
#include "Model.h"
//#include "Initializer.h"
//#include "Layer.h"
//#include "Neuron.h"

int main() {
  Initializer initializer(time(0));
  /*
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  std::string activationFunction = "Sigmoid";
  Neuron *neuron;
  auto fn =
      std::bind(&Initializer::randomNormal, initializer, std::placeholders::_1);
  neuron = new Neuron(3, activationFunction, fn);
  neuron->weights[0] = 3.0;
  neuron->weights[0] = 1.0;
  neuron->weights[0] = 1.0;

  generator.seed(time(0));
  for (int i = 0; i < 5; i++)
    std::cout << distribution(generator) << " ";
  std::cout << std::endl;

  std::normal_distribution<double> distribution2(0.0, 1.0);

  for (int i = 0; i < 5; i++)
    std::cout << distribution2(generator) << " ";
  std::cout << std::endl;
  // double output = neuron->getOutput(inputs);


  delete neuron;

  std::vector<double> inputs = {1.0, 1.5, 2.0, 2.0};
  Layer *layer = new DenseLayer(3, 4, "Sigmoid", "Random");
  layer->initializeWeights(initializer);
  std::cout << layer->totalParameters() << std::endl;
  std::vector<double> output = layer->getOutput(inputs);
  std::cout << "The output size is " << output.size() << std::endl;
  for (auto &n : output)
    std::cout << n << " ";
  std::cout << std::endl;
  delete layer;
  */

  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<DenseLayer>(20, 100, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(10, 0, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(5, 0, "Sigmoid"));
  Model model(layers);
  model.printModel();

  return 0;
}