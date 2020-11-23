//
// Created by juan on 20/11/20.
//

#include "NeuralNetwork.h"
#include "iostream"
#include "cmath"
using  namespace std;

/*
 * @param [in] n_layers number of intermediate layers
 * @param [in] n_neurons number of neurons in intermediate layers
 * */
NeuralNetwork::NeuralNetwork(int n_layers, int n_neurons, Dataset &dataset) {
    int n_init = dataset.x[0].size();                                       //first layer number of neurons (also number of specs of each entry)
    int n_exit = 1;                                                         //output layer number of neurons

    int n_training = dataset.y.size();                                      //number of training data

    // All Data
    x.assign(n_training, vec(n_init, 0));
    y.assign(n_training, 0);
    y_feat.assign(n_training, vec(1, 0));

    // Mini Batch data
    Mx.assign(n_training/10, vec(n_init, 0));
    My.assign(n_training/10, 0);
    My_feat.assign(n_training/10, vec(1, 0));

    //get data from dataset
    for (int i = 0; i<n_training; i++){
        y[i] = dataset.y[i];
        if (y[i] == 1){
            y_feat[i][0] = 1;
        }else{
            y_feat[i][0] = 0;
        }
        for (int j = 0; j<n_init; j++){
            x[i][j] = dataset.x[i][j];
        }
    }

    // check consistency of parameters
    if (n_layers <= 0){
        cout <<  "Number of Layers must be positive" << endl;
        return;
    }

    // build layers
    layers.push_back(new Layer(-1, n_init));
    for (int l = 0; l < n_layers; l++){
        if (l == 0)
            layers.push_back(new Layer(n_init, n_neurons));
        else
            layers.push_back(new Layer(n_neurons, n_neurons));
    }
    layers.push_back(new Layer(n_neurons, n_exit));
}

void NeuralNetwork::buildMiniBatch() {
    int n = Mx.size();
    int m = Mx[0].size();

    // create minibatch
    for (int i = 0; i<n; i++){
        int pos = getRandom(0, n-1);
        for (int j = 0; j<m; j++){
            Mx[i][j] = x[pos][j];
        }
        My[i] = y[pos];
    }

    // get mini batch of features
    CleanMatrix(My_feat);
    for (int i = 0; i<n; i++){
        if (My[i] == 1)
            My_feat[i][0] = 1;
        else
            My_feat[i][1] = 1;
    }
}

void NeuralNetwork::startTrainning(int epochs, double learning_rate) {
    if (learning_rate < 0 || learning_rate > 1){
        cout << "learning rate parameter out of bounds" << endl;
        //return;
    }
    int n = layers.size();
   /* for (int e = 0; e<epochs; e++){
        debug(e);
        showNeurons(e);
        Forward_Propagation(learning_rate);

        Output->updateWB(learning_rate/x.size());
        for (int i = n-1; i>0; i--){
            layers[i]->updateWB(learning_rate/x.size());
        }
    }
    */
    while (true){
        //debug(-1);
        Forward_Propagation(learning_rate);
        double error = getError();
        cout << "error: " << error << endl;
        if ( error < 0.1)
            break;

    }
}

void NeuralNetwork::debug(int e) {
    cout << "--------------------------------------------------------- SHOW WEIGHTS --------------------------------------------------------- " << endl;
    int n = layers.size();
    for (int l = 0; l< n; l++){
        cout << "EPOCH : " << e << " LAYER: " << l << endl;
        cout << layers[l]->weights << endl;
    }
}

void NeuralNetwork::showGradient(int e) {
    cout << "--------------------------------------------------------- SHOW GRADIENT --------------------------------------------------------- " << endl;
    int n = layers.size();
    for (int l = 0; l< n; l++){
        cout << "EPOCH : " << e << " LAYER: " << l+1 << endl;
        cout << layers[l]->weights << endl;
        cout << layers[l]->bias << endl;
    }
}

void NeuralNetwork::showNeurons(int e) {
    cout << "--------------------------------------------------------- SHOW NEURONS --------------------------------------------------------- " << endl;
    int n = layers.size();
    for (int l = 0; l<n; l++){
        cout << "EPOCH " << e << " NEURONS LAYER : " << l << endl;
        cout << layers[l]->neurons<< endl;
    }
}

void NeuralNetwork::cleanDeltas() {
    int l = layers.size();
    for (int k = 0; k<l; k++){
        CleanMatrix(layers[k]->weights_grad);
        CleanVector(layers[k]->bias_grad);
    }
}

void NeuralNetwork::feedInput(vec &input) {
    int n = layers[0]->neurons.size();
    if (input.size()!= n){
        cout << "ERROR: sizes dont match in Input" << endl;
        return;
    }
    for (int i = 0; i<n; i++){
        layers[0]->neurons[i] = input[i];
    }
}

void NeuralNetwork::Forward_Propagation(double learning_rate) {
    int n = layers.size();
    // for every x in the trainning sample
    for (int i = 0; i<x.size(); i++){
        feedInput(x[i]);
        // forward propagation
        for (int l = 1; l<layers.size(); l++){
            layers[l]->Forward_Propagation(layers[l-1]);
        }
        //showNeurons(0);
        Backward_Propagation(learning_rate, i);
        //showGradient(0);
    }
}

void NeuralNetwork::Backward_Propagation(double  learning_rate, int sample_i) {
    int n = layers.size();
    Layer *last = layers[n-1];
    for (int i = 0; i<layers[n-1]->neurons.size(); i++){
        last->delta[i] = (last->neurons[i] - y[sample_i]) * (last->neurons[i]) * (1.0 - last->neurons[i]);
    }
    // Backward Propagation
    for (int i = n-2; i>0; i--){
        layers[i]->Backward_Propagation(layers[i+1]);         // middle layers
    }
    for (int i = n-1; i>0; i--){
        layers[i]->updateWB(learning_rate, layers[i-1]);
    }
}

double NeuralNetwork::predict(vec &x) {
    int n = x.size();
    int l = layers.size();
    //debug(-1);
    feedInput(x);

    for (int i = 1; i<l; i++){
        layers[i]->Forward_Propagation(layers[i-1]);
    }
    double prediction = 0;
    //if (i_max == 0)
        //prediction = 1;
    prediction = round(layers[l-1]->neurons[0]);
    //cout << "Prediction :" << prediction << endl;
    return  prediction;
}


double NeuralNetwork::getError() {
    cout << " -------------------------------------------------- OUTPUT NEURONS ---------------------------------------------- " << endl;
    int n = x.size();
    double error = 0;
    for (int i = 0; i<n; i++){
        double prediction = predict(x[i]);
        if (fabs(prediction - y[i]) > 0.00001)
            error+= 1;

        //printf("prob 1: %0.6f \t\t\t ",Output->neurons[0]);
        //printf("prob 0: %0.6f \n",Output->neurons[1]);
    }
    return error/n;
}

int NeuralNetwork::getInputSize() {
    return x.size();
}

// FALTA utilizar otro vector para calcular errores porque vector acc_error debe ir acumulando el de cada ejemplo del set
// ACTUALIZAR PESOS Y BIAS para cada capa
// http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network
//http://neuralnetworksanddeeplearning.com/chap1.html
// https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4&ab_channel=3Blue1Brown