#ifndef NEURON_H
#define NEURON_H

#include "../eigen/Eigen/Eigen"
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Eigen;
class Neuron {
public:
  Neuron(int size, const std::string &activation,
         std::function<void(VectorXd &, double &)> initializer);
  double getOutput(VectorXd inputs);
  VectorXd weights;
  double bias;

protected:
  int size;
  std::string activation;
};

#endif