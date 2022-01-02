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
        Neuron();
        Neuron(int size, std::string actFunc,
                std::default_random_engine generator,
                std::normal_distribution<double> distribution);
        double getOutput(std::vector<double> inputs);
        std::vector<double> weights;

    protected:
        int size;
        std::string activation; 
};

#endif