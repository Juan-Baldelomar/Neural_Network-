//
// Created by juan on 20/11/20.
//

#include "Neuron.h"
#include "math.h"

int hard_limit(double x){
    return x<0? 0 : 1;
}

double sigmoid(double x){
    return  1/(1+pow(M_E, x));
}

double sigmoid_prime(double x){
    return sigmoid(x)*(1-sigmoid(x));
}


Neuron::Neuron(int n) {
    n_weights = n;
    weights = new double[n];
}

Neuron::~Neuron() {
    delete (weights);
}

void Neuron::activation() {
    if (activation_code == 0){
        activation_value = hard_limit(activation_value);
    }else if(activation_code==1){
        activation_value = sigmoid(activation_value);
    }else{
        activation_value = hard_limit(activation_value);
    }
}


