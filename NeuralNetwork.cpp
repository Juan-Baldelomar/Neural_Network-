//
// Created by juan on 20/11/20.
//

#include "NeuralNetwork.h"
#include "iostream"
using  namespace std;

/*
 * @param [in] n_layers number of intermediate layers
 * @param [in] n_neurons number of neurons in intermediate layers
 * */
NeuralNetwork::NeuralNetwork(int n_layers, int n_neurons, Dataset &dataset) {
    int n_init = dataset.x[0].size();                                       //first layer number of neurons (also number of specs of each entry)
    int n_exit = 2;                                                         //output layer number of neurons
    int n_training = dataset.y.size();                                      //number of training data

    this->n_layers = n_layers;
    x = new double*[n_training];
    y = new double[n_training];

    //get data from dataset
    for (int i = 0; i<n_training; i++){
        x[i] = new double[n_init];
        y[i] = dataset.y[i];
        for (int j = 0; j<n_init; j++){
            x[i][j] = dataset.x[i][j];
        }
    }


    if (n_layers>0 && n_neurons<= 0){
        cout <<  "Number of Neurons invalid" << endl;
        return;
    }

    if (n_layers > 0){
        Input = new Layer(-1, n_neurons, n_init);
        Output = new Layer(n_neurons, -1, n_exit);
    } else{                                                                 //if there are no intermediate layers
        Input = new Layer(-1, n_exit, n_init);
        Output = new Layer(n_init, -1, n_exit);
    }


}