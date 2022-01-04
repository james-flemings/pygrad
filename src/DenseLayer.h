#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Layer.h"

class DenseLayer : public Layer {
public:
  DenseLayer();
  DenseLayer(int units, int inputSize = 0,
             const std::string activation = std::string(),
             const std::string initialization = std::string());
  std::any getOutputImpl(std::any inputs) override;
  /*
  std::unique_ptr<Layer> cloneBase() override {
    return std::make_unique<Layer>(*this);
  }
  */
};

#endif