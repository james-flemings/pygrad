#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <random>
#include <time.h>
#include <vector>

class Initializer {
public:
  Initializer(int seed);
  void randomNormal(std::vector<double> &weights);

protected:
  std::default_random_engine generator;
};

#endif