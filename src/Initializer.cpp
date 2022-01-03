#include "Initializer.h"

Initializer::Initializer(int seed) { generator.seed(seed); }

void Initializer::randomNormal(std::vector<double> weights) {
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (auto &w : weights)
    w = distribution(this->generator);
}