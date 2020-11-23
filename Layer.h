//
// Created by juan on 20/11/20.
//

#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H
#include "Neuron.h"
#include "vector"
#include "iostream"
using namespace std;

typedef vector<double> vec;
typedef vector<vec> matrix;

void Resta_Vector(vec &v1, vec &v2, vec &res);
void Suma_Vector(vec &v1, vec &v2, vec &res);
void Hadamard_Product(vec &v1, vec &v2, vec &res);
double sigmoid_prime(double x);
void CleanMatrix(matrix &A);
void CleanVector(vec &v);

ostream &operator<<(ostream &os, const vector< vector<double> > &M);
ostream &operator<<(ostream &os, const vector<double> &V);

class Layer {
private:
    int inputSize, outputSize;
    void Activation();

public:
    vec neurons, z, acc_error, bias, bias_grad, delta;
    matrix weights, weights_grad;
    Layer(int inputSize, int outputSize, int n);
    int getInputSize();
    int getOutputSize();
    int get_neuronsCount();
    void Forward_Propagation(Layer *prev);
    void Backward_Propagation(matrix & nxt_weights, vec & delta_nxLayer, vec & neurons_prevLayer);
    void gradient_calculation(Layer *prev);
    void addWeightsGrad(vec &a_x);
    void updateWB(double learning_rate);
    void Error_Activation();
};


#endif //NEURAL_NETWORK_LAYER_H
