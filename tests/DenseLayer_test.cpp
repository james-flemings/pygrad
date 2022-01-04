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

TEST_F(DenseLayerTest, ParametersAssertions) {
  EXPECT_EQ(layer_->totalParameters(), UNITS * (INPUTSIZE + 1))
      << "Layer does not contain correct number of parameters";
}

TEST_F(DenseLayerTest, OutputShapeAssertions) {
  std::vector<double> input = {1.0, 2.0, 2.0, 3.0};
  std::vector<double> output = layer_->getOutput(input);
  EXPECT_EQ(output.size(), UNITS)
      << "Layer output shape does not match number of units";
  for (auto &o : output) {
    EXPECT_LT(o, 1) << "Output larger than 1";
    EXPECT_GT(o, 0) << "Output less than 0";
  }
}

TEST_F(DenseLayerTest, ArgumentsAssertions) {
  EXPECT_THROW({ DenseLayer layer(10, 0, "Foo"); }, std::invalid_argument)
      << "Failed error checking for activation function";
  EXPECT_THROW({ DenseLayer layer(10, 0, "Sigmoid", "Foo"); },
               std::invalid_argument)
      << "Failed error checking for initialization function";
  EXPECT_THROW({ DenseLayer layer(0); }, std::invalid_argument)
      << "Failed error checking for units";
  DenseLayer layer(10);
  const std::string activation = "None";
  const std::string initialization = "Random";

  EXPECT_EQ(layer.getUnits(), 10) << "Incorrect Units";
  EXPECT_EQ(layer.getActivation(), activation)
      << "Incorrect default activation";
  EXPECT_EQ(layer.getInitialization(), initialization)
      << "Incorrect default initialization";
}