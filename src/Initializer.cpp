#include "Initializer.h"

Initializer::Initializer(int seed = -1) {
  if (seed != -1)
    this->generator.seed(seed);
  else
    this->generator.seed(time(0));
}

void Initializer::randomNormal(std::vector<double> weights) {
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (auto &w : weights)
    w = distribution(this->generator);
}