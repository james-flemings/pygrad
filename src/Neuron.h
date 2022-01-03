#ifndef NEURON_H
#define NEURON_H

#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class Neuron {
public:
  Neuron(int size, const std::string &activation,
         std::function<void(std::vector<double>)> initializer);
  double getOutput(std::vector<double> inputs);
  std::vector<double> weights;

protected:
  int size;
  std::string activation;
};

#endif