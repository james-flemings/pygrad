#include "../src/DenseLayer.h"
#include "../src/Layer.h"
#include <gtest/gtest.h>

const int UNITS = 3;
const int INPUTSIZE = 4;

class DenseLayerTest : public ::testing::Test {
protected:
  void SetUp() override {
    Initializer initializer(time(0));
    layer_ = new DenseLayer(UNITS, INPUTSIZE, "Sigmoid", "Random");
    layer_->initializeWeights(initializer);
  }
  void TearDown() override {
    delete layer_;
    layer_ = nullptr;
  }
  DenseLayer *layer_;
};

TEST_F(DenseLayerTest, NeuronAssertions) {
  EXPECT_EQ(layer_->totalParameters(), UNITS * (INPUTSIZE + 1))
      << "Layer does not contain correct number of parameters";
}