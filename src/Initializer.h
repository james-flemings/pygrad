#ifndef INITIALIZER_H
#define INITIALIZER_H

#include "../eigen/Eigen/Eigen"
#include <random>
#include <time.h>
#include <vector>

using namespace Eigen;

class Initializer {
public:
  Initializer(int seed);
  void randomNormal(VectorXd &weights, double &bias);

protected:
  std::default_random_engine generator;
};

#endif