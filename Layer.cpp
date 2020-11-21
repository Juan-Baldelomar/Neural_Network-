//
// Created by juan on 20/11/20.
//

#include "Layer.h"
/*
 * @param [in] inputSize number of inputs per neuron in layer
 * @param [in] outputSize number of outputs per neuron in layer (number of neurons in next layer)
 * @param [in] n number of neurons in layer
 */
Layer::Layer(int inputSize, int outputSize, int n) {
    n_neurons = n;
    neurons = new Neuron*[n];
    for (int i = 0; i<n; i++){
        neurons[i] = new Neuron(inputSize);
    }
}

int Layer::getInputSize() {
    return inputSize;
}

int Layer::getOutputSize() {
    return outputSize;
}

int Layer::get_neuronsCount() {
    return n_neurons;
}