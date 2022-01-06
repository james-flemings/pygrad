#include "../src/Model.h"
#include <gtest/gtest.h>

const int UNITS = 3;
const int INPUTSIZE = 4;

class ModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::make_unique<DenseLayer>(5, 3, "Sigmoid"));
    layers.push_back(std::make_unique<DenseLayer>(2, 0, "Sigmoid"));
    model = std::make_unique<Model>(layers);
  }
  // void TearDown() override {}
  std::unique_ptr<Model> model;
};

TEST_F(ModelTest, ParametersAssertions) {
  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<DenseLayer>(10, 10, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(10, 0, "Sigmoid"));

  Model model(layers);
  EXPECT_EQ(model.totalParameters(), 10 * 11 + 10 * 11)
      << "Layer does not contain correct number of parameters";
}

TEST_F(ModelTest, ForwardPassAssertions) {
  VectorXd inputs{{1.0, 1.0, 1.0}};
  std::vector<VectorXd> activations;
  model->forwardPass(activations, inputs);
  EXPECT_EQ(activations.size(), 3)
      << "Number of activations doesn't equal number of layers";

  EXPECT_EQ(activations[0].size(), inputs.size())
      << "Size of activations doesn't equal number of units for each layer";
  for (int i = 1; i < 3; i++) {
    EXPECT_EQ(activations[i].size(), model->layers[i - 1]->getUnits())
        << "Size of activations doesn't equal number of units for each layer";
  }
}