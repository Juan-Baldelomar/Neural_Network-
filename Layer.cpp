//
// Created by juan on 20/11/20.
//

#include "Layer.h"
#include "math.h"
#include "cstdlib"
#include <bits/stdc++.h>

// utilities
double sigmoid(double x){
    return  1.0/(1.0+exp(-x));
}

/* ----------------------------------------------------------------- SOBRECARGA OPERADORES ----------------------------------------------------------------------- */
ostream &operator<<(ostream &os, const vector< vector<double> > &M)
{
    for(int i = 0; i < M.size(); i++)
    {
        for(auto x:M[i])
        {
            os << setw(15) << x << setw(15);
        }
        os <<endl;
    }
    return os;
}
ostream &operator<<(ostream &os, const vector<double> &V)
{
    for(auto x:V)
        os << x << " ";
    os <<endl;
    return os;
}

vector<double> operator+(vec&v1, vec&v2){
    int n = v1.size();
    vec v;
    for (int i = 0; i<n; i++){
        v.push_back(v1[i] + v2[i]);
    }
    return v;
}

vec operator*(matrix &mat, vec &v){
    int n = mat.size();
    int m = mat[0].size();
    vec res;
    for (int i = 0; i<n; i++){
        double acc = 0;
        for (int j = 0; j<m; j++){
            acc+= mat[i][j]*v[j];
        }
        res.push_back(acc);
    }
    return res;
}

vec operator*(vec &v1, vec &v2){
    int n = v1.size();
    vec res;
    for (int i = 0; i<n; i++){
        res.push_back(v1[i]*v2[i]);
    }
    return res;
}

matrix operator!(matrix &mat){
    int n = mat.size();
    int m = mat[0].size();
    matrix res(m, vector<double>(n, 0));
    for (int i = 0; i<n; i++)
        for (int j = 0; j<m; j++)
            res[j][i] = mat[i][j];

    return res;
}

int Layer::getInputSize() {
    return inputSize;
}

/*---------------------------------------- CONSTRUCTOR -----------------------------------------------*/

Layer::Layer(int inputSize, int n) {
    this->inputSize = inputSize;
    neurons.assign(n, 0);
    bias.assign(n, 1);
    z.assign(n, 0);
    delta.assign(n, 0);

    if (inputSize!= -1){
        weights.assign(n, vec(inputSize, 0));
    }

    // assign random values to weights and bias
    for (int i = 0; i<n; i++){
        bias[i] = 1.0 * rand()/(RAND_MAX);
        for (int j = 0; j<inputSize; j++){
            weights[i][j] = 1.0* rand()/(RAND_MAX);
        }
    }
}

/* ---------------------------------------------------------------- TRAINNING -----------------------------------------------------------------------*/

void Layer::Activation() {
    int n = neurons.size();
    for (int i = 0; i<n; i++){
        neurons[i] = sigmoid(z[i]);
    }
}

void Layer::Forward_Propagation(Layer *prev) {
    z = weights * prev->neurons;
    z = bias + z;
    Activation();
}

void Layer::Backward_Propagation(Layer *next) {
    vec sigm_prime;
    matrix mat = !next->weights;                            // transpose weights
    vec tmp = mat * next->delta;                            // matrix * vec
    for (int i = 0; i < neurons.size(); i++)
        sigm_prime.push_back(neurons[i] * (1.0 - neurons[i]));

    delta = sigm_prime * tmp;                               //hardamar product
}

void Layer::updateWB(double learning_rate, Layer *prev) {
    int n = weights.size();
    int m = weights[0].size();

    for (int i = 0; i<n; i++){
        bias[i] += -learning_rate * delta[i];
        for (int  j = 0; j<m; j++){
            weights[i][j] += - learning_rate * delta[i] * prev->neurons[j] ;
        }
    }
}