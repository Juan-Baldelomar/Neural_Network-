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
    Layer *Input;
    Layer *Output;
    vector<Layer*>layers;
    vector<vec>x, y_feat, Mx, My_feat;
    vec y, My;

public:
    NeuralNetwork(int n_layers, int n_neurons, Dataset &dataset);
    void startTrainning(int epochs, double learning_rate);
    void buildMiniBatch();
    void Forward_Propagation(double learning_rate);
    void Backward_Propagation(double learning_rate, int sample_i);
    double predict(vec &x);
    double getError();
    void debug(int e);
    void showNeurons(int e);
    void showGradient(int e);
    void cleanDeltas();

    int getInputSize();
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
