#include "Initializer.h"

#include <iostream>

Initializer::Initializer(int seed) { generator.seed(seed); }

void Initializer::randomNormal(VectorXd &weights, double &bias) {
  std::random_device rd;
  generator.seed(rd());
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (auto &w : weights) {
    w = distribution(this->generator);
  }
  bias = distribution(this->generator);
}