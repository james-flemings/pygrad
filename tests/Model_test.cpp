#include "../src/DenseLayer.h"
#include "../src/Model.h"
#include <gtest/gtest.h>

TEST(ModelTest, ParametersAssertions) {
  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<DenseLayer>(10, 10, "Sigmoid"));
  layers.push_back(std::make_unique<DenseLayer>(10, 0, "Sigmoid"));

  Model model(layers);
  EXPECT_EQ(model.totalParameters(), 10 * 11 + 10 * 11)
      << "Layer does not contain correct number of parameters";
}