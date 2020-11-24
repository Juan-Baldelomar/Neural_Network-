//
// Created by juan on 20/11/20.
//

#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H
#include "vector"
#include "iostream"
using namespace std;

typedef vector<double> vec;
typedef vector<vec> matrix;


ostream &operator<<(ostream &os, const vector< vector<double> > &M);
ostream &operator<<(ostream &os, const vector<double> &V);

class Layer {
private:
    int inputSize;
    void Activation();

public:
    vec neurons, z, bias, delta;
    matrix weights;
    Layer(int inputSize, int n);
    ~Layer();
    int getInputSize();
    void Forward_Propagation(Layer *prev);
    void Backward_Propagation(Layer *next);
    void updateWB(double learning_rate, Layer* prev);
};


#endif //NEURAL_NETWORK_LAYER_H
