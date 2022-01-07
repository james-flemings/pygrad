#include "Model.h"

Model::Model(std::vector<std::unique_ptr<Layer>> &layers) {
  // First layer must contain the input size
  if (layers[0]->inputSize == 0)
    throw std::invalid_argument("First layer must contain input size");

  Initializer initializer(time(0));
  int prevInputSize = 0;

  this->layers.resize(layers.size());
  std::move(layers.begin(), layers.end(), this->layers.begin());

  for (auto &layer : this->layers) {
    layer->inputSize = layer->inputSize == 0 ? prevInputSize : layer->inputSize;
    prevInputSize = layer->getUnits();
    layer->initializeWeights(initializer);
  }
}

int Model::totalParameters() {
  int total = 0;
  for (auto &layer : this->layers)
    total += layer->totalParameters();
  return total;
}

void Model::summary() {
  std::string row = "---------------------------";
  std::cout << "\nLayer \t Units \t Parameters \n";
  std::cout << row << std::endl;
  std::cout << "Input \t " << layers[0]->inputSize << " \t "
            << "__ \n";
  for (auto &layer : this->layers) {
    std::cout << row << std::endl;
    std::cout << "Dense \t " << layer->getUnits() << " \t "
              << layer->totalParameters() << std::endl;
  }
  std::cout << row << std::endl;
  std::cout << "Total Prameters: " << this->totalParameters() << "\n\n";
}

results Model::train(const data &training_data, const int epochs, int batchSize,
                     double lr, double reg_term, const data &validation_data) {
  int n = training_data.size();
  std::vector<double> eval_cost, eval_accuracy, train_cost, train_accurracy;
  std::vector<data> miniBatches;
  for (int i = 0; i < epochs; i++) {
    std::random_shuffle(training_data.begin(), training_data.end());
    for (int j = 0; j < n; i += batchSize) {
      this->updateMiniBatch(
          data(&training_data[j], &training_data[j + batchSize]), lr, reg_term,
          n);
    }
    std::cout << "Epoch " << i << " training complete\n";
  }
}

void Model::updateMiniBatch(const data &miniBatch, double lr, double reg_term,
                            int n) {
  int size = this->layers.size();
  std::vector<VectorXd> nabla_b(size), delta_nabla_b;
  std::vector<MatrixXd> nabla_w(size), delta_nabla_w;
  VectorXd input, labels, bias;
  MatrixXd weights;

  for (int i = 0; i < size; i++) {
    nabla_b[i].resize(this->layers[i]->getUnits());
    nabla_w[i].resize(this->layers[i]->getUnits(), this->layers[i]->inputSize);
  }

  for (auto &mb : miniBatch) {
    auto [input, labels] = mb;
    auto [delta_nabla_b, delta_nabla_w] = backProp(input, labels);
    for (int i = 0; i < size; i++) {
      nabla_b[i] += delta_nabla_b[i];
      nabla_w[i] += delta_nabla_w[i];
    }
  }
  for (int i = 0; i < size; i++) {
    weights = (1 - lr * reg_term / n) * this->layers[i]->getWeights().array() -
              (lr / miniBatch.size()) * nabla_w[i].array();
    bias = this->layers[i]->getBias().array() -
           (lr / miniBatch.size()) * nabla_b[i].array();
    this->layers[i]->setWeights(weights);
    this->layers[i]->setBias(bias);
  }
}

back_prop_type Model::backProp(const VectorXd &input, const VectorXd &label) {
  int n = this->layers.size();
  std::vector<VectorXd> nabla_b(n), activations;
  std::vector<MatrixXd> nabla_w(n);

  forwardPass(activations, input);

  // backward pass
  VectorXd del =
      this->delta(this->layers.back()->sigmoidPrime(activations.back()),
                  activations.back(), label);
  nabla_b.back() = del;
  nabla_w.back() = del * (activations[n - 2]).transpose();

  VectorXd sp;
  for (int i = n - 2; i > 0; i--) {
    sp = this->layers[i]->sigmoidPrime(activations[i]);
    del = (this->layers[i + 1]->getWeights().transpose() * del).array() *
          sp.array();
    nabla_b[i] = del;
    nabla_w[i] = del * activations[i - 1].transpose();
  }
  return std::make_tuple(nabla_b, nabla_w);
}

void Model::forwardPass(std::vector<VectorXd> &activations,
                        const VectorXd &input) {
  // Forward pass
  VectorXd activation = input;
  VectorXd output;
  activations.push_back(activation);
  for (auto &layer : this->layers) {
    output = layer->getOutput(activation);
    activations.push_back(output);
    activation = output;
  }
}

double Model::cost(const VectorXd &activations, const VectorXd &labels) {
  /*
  Quadratic cost function
  */
  return 0.5 * (activations - labels).squaredNorm();
}

VectorXd Model::delta(const VectorXd sp, const VectorXd &activations,
                      const VectorXd &labels) {
  /*
  Gradient of cost function with respect to previous activations.
  */
  return (activations - labels).array() * sp.array();
}