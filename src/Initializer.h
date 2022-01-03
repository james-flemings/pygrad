#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <random>
#include <time.h>
#include <vector>

class Initializer {
public:
  Initializer(int seed = -1);
  void randomNormal(std::vector<double> weights);

protected:
  std::default_random_engine generator;
};

#endif