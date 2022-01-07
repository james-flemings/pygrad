#ifndef MODEL_H
#define MODEL_H

#include "DenseLayer.h"
#include "Layer.h"
#include <algorithm>
#include <memory>
#include <tuple>

/*
After training, return a tuple of four vectors (loss_v, acc_v, loss_t, acc_t):
loss_v : validation loss
acc_v  : validation accuracy
loss_t : training loss
acc_t  : training accuracy
*/
typedef std::tuple<std::vector<double>, std::vector<double>,
                   std::vector<double>, std::vector<double>>
    results;

/*
Type for training data, which is a vector of tuples where each tuple contains
two vectors [(x, y), (x, y), ..., (x, y)]
x: vector of inputs
y: vector of outputs
*/
typedef std::vector<std::tuple<VectorXd, VectorXd>> data;
typedef std::tuple<std::vector<VectorXd>, std::vector<MatrixXd>> back_prop_type;
class Model {
public:
  Model();
  Model(std::vector<std::unique_ptr<Layer>> &layers);
  int totalParameters();
  void summary();
  results train(const data &training_data, const int epochs = 20,
                int batchSize = 32, double lr = 0.1,
                double regularizer_term = 0,
                const data &validation_data = data());
  void updateMiniBatch(const data &miniBatch, double lr,
                       double regularizer_term, int n);
  back_prop_type backProp(const VectorXd &input, const VectorXd &label);
  void forwardPass(std::vector<VectorXd> &activations, const VectorXd &input);
  double cost(const VectorXd &activations, const VectorXd &labels);
  VectorXd delta(const VectorXd sp, const VectorXd &activations,
                 const VectorXd &labels);
  std::vector<std::unique_ptr<Layer>> layers;

protected:
  std::string loss, optimizer;
  double lr, reg_term;
  int batchSize;
};

#endif