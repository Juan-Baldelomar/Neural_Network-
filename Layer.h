//
// Created by juan on 20/11/20.
//

#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H
#include "Neuron.h"

class Layer {
private:
    int n_neurons, inputSize, outputSize;
    Neuron **neurons;
public:
    Layer(int inputSize, int outputSize, int n);
    int getInputSize();
    int getOutputSize();
    int get_neuronsCount();
};


#endif //NEURAL_NETWORK_LAYER_H
