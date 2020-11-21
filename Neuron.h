//
// Created by juan on 20/11/20.
//

#ifndef NEURAL_NETWORK_NEURON_H
#define NEURAL_NETWORK_NEURON_H


class Neuron {
    int activation_code;
    int n_weights;
public:
    double activation_value;
    double desired_value;
    double *weights;
    double bias;
    Neuron(int n);
    ~Neuron();
    void activation();
};


#endif //NEURAL_NETWORK_NEURON_H
