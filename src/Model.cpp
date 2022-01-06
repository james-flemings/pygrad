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

back_prop_type Model::backProp(const VectorXd &input, const VectorXd &label) {
  std::vector<VectorXd> activations;
  int n = this->layers.size();
  std::vector<VectorXd> nabla_b(n);
  std::vector<MatrixXd> nabla_w(n);

  forwardPass(activations, input);

  // backward pass
  VectorXd del =
      this->delta(this->layers.back()->sigmoidPrime(activations.back()),
                  activations.back(), label);
  nabla_b.back() = del;
  nabla_w.back() = del * (activations[n - 2]).transpose();

  VectorXd sp;
  for (int i = n - 2; i > 0; i++) {
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