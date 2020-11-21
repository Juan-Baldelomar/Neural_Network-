//
// Created by juan on 20/11/20.
//

#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H
#include "Layer.h"
#include "Dataset.h"

class NeuralNetwork {
    Layer *Input;
    Layer *Output;
    Layer **layers;
    double **x;
    double *y;
    int n_layers;

public:
    NeuralNetwork(int n_layers, int n_neurons, Dataset &dataset);
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
