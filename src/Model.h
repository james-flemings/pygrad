#ifndef MODEL_H
#define MODEL_H

#include "DenseLayer.h"
#include "Layer.h"
#include <algorithm>
#include <memory>

class Model {
public:
  Model();
  Model(std::vector<std::unique_ptr<Layer>> &layers);
  int totalParameters();

protected:
  std::vector<std::unique_ptr<Layer>> layers;
};

#endif