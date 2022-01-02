#ifndef NEURON_H
#define NEURON_H

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class Neuron {
public:
  Neuron();
  Neuron(int size, std::string actFunc, std::default_random_engine generator,
         std::normal_distribution<double> distribution);
  double getOutput(std::vector<double> inputs);
  std::vector<double> weights;

protected:
  int size;
  std::string activation;
};

#endif