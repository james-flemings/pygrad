#include "../src/Initializer.h"
#include "../src/Neuron.h"
#include <gtest/gtest.h>

const int SIZE = 3;

class NeuronTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::default_random_engine generator(time(0));
    std::normal_distribution<double> distribution(0.0, 1.0);
    std::string activation = "Sigmoid";
    Initializer initialize(time(0));
    auto fn = std::bind(&Initializer::randomNormal, initialize,
                        std::placeholders::_1);
    n_ = new Neuron(SIZE, activation, fn);
    n_->weights[0] = 3.0;
    n_->weights[1] = 1.0;
    n_->weights[2] = 1.0;
  }
  void TearDown() override {
    delete n_;
    n_ = nullptr;
  }
  Neuron *n_;
};

TEST_F(NeuronTest, WeightsAssertions) {
  EXPECT_EQ(n_->weights.size(), SIZE + 1)
      << "Neuron did not create correct number of weights";

  std::vector<double> inputs = {1.0, 1.0};
  double output = n_->getOutput(inputs);
  double expected = 1 / (1 + exp(-5));
  EXPECT_EQ(output, expected) << "Neuron did not output correctly";
}