#include "Neuron.h"

Neuron::Neuron(int size, std::string actFunc,  
                std::default_random_engine generator,
                std::normal_distribution<double> distribution) {
    /*
    Initilize Neuron by determining activation funciton and intializing the weights
    TODO: Make initializer modular so we can use different inialization schemes
    Right now weights (and bias) are initialized by randomly selecting a value
    from a normal distribution
    */
    // If user requests unimplemented activation function, throw exception
    // Might need to move this to Layers class once in development
    if (actFunc.compare("Sigmoid"))
            throw std::invalid_argument("Invalid activation function");
    this->activationFunction = actFunc;
    this->size = size;
    for (int i = 0; i < this->size; i++){
        this->weights.push_back(distribution(generator));
    }
}

double Neuron::getOutput(std::vector<double> inputs){
    /*
    Calculate output of a neuron given the inputs with corresponding weights (and bias)
    */
    double output = 0.0;
    if (!this->activationFunction.compare("Sigmoid")){
        double product = std::inner_product(this->weights.begin()+1, this->weights.end(), inputs.begin(), this->weights[0]);
        output =  1 / (1 + exp(-product));
    }
    else {
        throw std::domain_error("Invalid activation function in get output. This should not occur.");
    }
    return output;
}
