//
// Created by juan on 20/11/20.
//

#include "Layer.h"
#include "math.h"
#include "cstdlib"
#include <bits/stdc++.h>


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


int hard_limit(double x){
    return x<0? 0 : 1;
}

double sigmoid(double x){
    return  1.0/(1.0+exp(-x));
}

double sigmoid_prime(double x){
    return sigmoid(x)*(1-sigmoid(x));
}

void Resta_Vector(vec &v1, vec &v2, vec &res){
    int n = v1.size();
    for (int i = 0; i<n; i++){
        res[i] = v1[i] - v2[i];
    }
}

/*
vector<double> operator-(vec&v1, vec&v2){
    int n = v1.size();
    vector<double> *v = new vector<double>(n, 0);
    for (int i = 0; i<n; i++){
        (*v)[i] = v1[i] - v2[i];
    }
    return v;
}
*/

void Suma_Vector(vec &v1, vec &v2, vec &res){
    int n = v1.size();
    for (int i = 0; i<n; i++){
        res[i] = v2[i] + v1[i];
    }
}

void Transpose(matrix &mat, matrix &res){
    int n = mat.size();
    int m = mat[0].size();
    res.assign(m, vector<double>(n, 0));

    for (int i = 0; i<n; i++)
        for (int j = 0; j<m; j++)
            res[j][i] = mat[i][j];
}

void Matrix_Vector_Mult(matrix &mat, vec &v, vec &res){
    int n = mat.size();
    int m = mat[0].size();
    for (int i = 0; i<n; i++){
        double acc = 0;
        for (int j = 0; j<m; j++){
            acc+= mat[i][j]*v[j];
        }
        res[i] = acc;
    }
}

void CleanMatrix(matrix &A){
    int n = A.size();
    int m = A[0].size();
    for (int i = 0; i<n; i++)
        for (int j = 0; j<m; j++)
            A[i][j] = 0;
}

void CleanVector(vec &v){
    int n = v.size();
    for (int i =0; i<n; i++)
        v[i] = 0;
}

void Hadamard_Product(vec &v1, vec &v2, vec &res){
    int n = v1.size();
    for (int i = 0; i<n; i++){
        res[i] = v1[i]*v2[i];
    }
}

int Layer::getInputSize() {
    return inputSize;
}

int Layer::get_neuronsCount() {
    return neurons.size();
}

Layer::Layer(int inputSize, int n) {
    /*if (inputSize == -1)
        n = n-1;*/
    this->inputSize = inputSize;
    neurons.assign(n, 0); bias.assign(n, 1); acc_error.assign(n, 0);
    bias_grad.assign(n, 0); z.assign(n, 0); delta.assign(n, 0);

    if (inputSize!= -1){
        weights.assign(n, vec(inputSize, 0));
        weights_grad.assign(n, vec(inputSize, 0));
    }

    // assign random values to weights and bias
    for (int i = 0; i<n; i++){
        bias[i] = 1.0 * rand()/(RAND_MAX);
        for (int j = 0; j<inputSize; j++){
            weights[i][j] = 1.0* rand()/(RAND_MAX);
        }
    }
   // if (inputSize != -1)
        //neurons[n] = 1;
}

void Layer::Activation() {
    int n = neurons.size();
    for (int i = 0; i<n; i++){
        neurons[i] = sigmoid(z[i]);
    }
}

void Layer::Forward_Propagation(Layer *prev) {
    Matrix_Vector_Mult(weights, prev->neurons, z);
    Suma_Vector(z, bias, z);
    Activation();
}

// delta_nxLayer = delta_{l+1}
// neurons_prevLayer = neurons_{l-1}
void Layer::Backward_Propagation(Layer *next) {
    matrix mat;
    Transpose(next->weights, mat);

    vec tmp(mat.size(), 0);
    vec tmp_error(z.size(), 0);

    Matrix_Vector_Mult(mat, next->delta, tmp);
    for (int i = 0; i<tmp_error.size(); i++)
        tmp_error[i] = neurons[i]* (1.0 - neurons[i]);
        //tmp_error[i] = sigmoid_prime(z[i]);

    Hadamard_Product(tmp_error, tmp, delta);
    //addWeightsGrad(neurons_prevLayer);
}

void Layer::addWeightsGrad(vec &a_x) {
    int n = delta.size();
    int m = a_x.size();
    matrix mat(n, vec(m, 0));

    // vectors mult
    for (int i = 0; i<n; i++){
        for (int j = 0; j<m; j++){
            mat[i][j] = delta[i]*a_x[j];
        }
    }
    for (int i = 0; i<n; i++){
        bias_grad[i] += delta[i];
        for (int j = 0; j<m; j++){
            weights_grad[i][j] += mat[i][j];
        }
    }
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