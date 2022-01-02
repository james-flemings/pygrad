#include "Neuron.h"

Neuron::Neuron(int size, std::string activationFunction, int seed=time(0), 
                std::default_random_engine generator,
                std::normal_distribution<double> distribution) {
    if (!activationFunction.compare("Sigmoid"))
            throw std::invalid_argument("Invalid activation function");
    this->activationFunction = activationFunction;
    this->size = size;
    for (int i = 0; i < this->size; i++){
        this->weights.push_back(distribution(generator));
    }
}

double Neuron::getOutput(std::vector<double> inputs){
    double output = 0.0;
    double sum = 0.0;
    if (!this->activationFunction.compare("Sigmoid")){
        for (auto& n : this->weights)
            sum += n * 1;
        sum = std::accumulate(this->weights.begin(), this->weights.end(), 
                                    decltype(this->weights)::value_type(0));
        output =  1 / (1 + exp(-sum));
    }
    return output;
}
