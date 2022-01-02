#ifndef NEURON_H
#define NEURON_H

#include <string>
#include <random>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <numeric>

class Neuron
{
    public:
        Neuron(int size, std::string activationFunction, int seed,
                std::default_random_engine generator,
                std::normal_distribution<double> distribution);
        double getOutput(std::vector<double> inputs);

    private:
        int size;
        std::string activationFunction; 
        std::vector<double> weights;
};

#endif