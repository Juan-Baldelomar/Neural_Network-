//
// Created by juan on 20/11/20.
//

#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H
#include "Layer.h"
#include "Dataset.h"
#include "vector"

typedef vector<double> vec;

class NeuralNetwork {
    vector<Layer*>layers;
    vector<vec>x, y;

public:
    NeuralNetwork(int n_layers, int n_neurons, Dataset &dataset);
    void startTrainning(int epochs, double learning_rate);
    void Forward_Propagation(vec &input);
    void Backward_Propagation(double learning_rate, vec &expected);
    double predict(vec &x);
    double getError();
    void debug(int e);
    void showNeurons(int e);
    void showGradient(int e);
    void feedInput(vec &input);
    int getInputSize();
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
