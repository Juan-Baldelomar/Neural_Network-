//
// Created by juan on 20/11/20.
//

#include "NeuralNetwork.h"
#include "iostream"
#include "cmath"

using namespace std;

NeuralNetwork::~NeuralNetwork() {
    for (unsigned int i = 0; i<layers.size(); i++){
        delete (layers[i]);
    }
}

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
    y.assign(n_training, vec(n_exit, 0));

    //get data from dataset
    for (int i = 0; i < n_training; i++) {
        y[i][0] = dataset.y[i];
        for (int j = 0; j < n_init; j++) {
            x[i][j] = dataset.x[i][j];
        }
    }
    // check consistency of parameters
    if (n_layers <= 0) {
        cout << "Number of Layers must be positive" << endl;
        return;
    }

    // build layers
    layers.push_back(new Layer(-1, n_init));
    for (int l = 0; l < n_layers; l++) {
        if (l == 0)
            layers.push_back(new Layer(n_init, n_neurons));
        else
            layers.push_back(new Layer(n_neurons, n_neurons));
    }
    layers.push_back(new Layer(n_neurons, n_exit));
}


void NeuralNetwork::startTrainning(int epochs, double learning_rate, double tol) {
    if (learning_rate < 0 || learning_rate > 1) {
        cout << "learning rate parameter out of bounds" << endl;                //verify learning rate is within bounds
        return;
    }

    int e = 1;
    double error;
    while (true) {
        for (unsigned int i = 0; i < x.size(); i++) {
            Forward_Propagation(x[i]);
            Backward_Propagation(learning_rate, y[i]);
        }

        // verify error in prediction
        error = getError();
        cout << "error: " << error << endl;
        if (error < tol || ++e >= epochs)
            break;                                                              //stop criteria
    }

    //statistics

    cout << " --------------------------------------------- STATISTICS ------------------------------------------------ " <<endl;
    if (error >= tol)
        cout << "  WARNING : NO CONVERGENCIA  !! " << endl;
    cout << "TOLERANCIA: " << tol << endl;
    cout << "HIDDEN LAYERS: " << layers.size() - 2 << endl;
    cout << "Neuronas: " << layers[1]->neurons.size() << endl;
    cout << "EPOCAS: " << e << endl;
    cout << "ERROR: " << error << endl;
}

// feed input to input layer
void NeuralNetwork::feedInput(vec &input) {
    unsigned int n = layers[0]->neurons.size();
    if (input.size() != n) {
        cout << "ERROR: sizes dont match in Input" << endl;
        return;
    }
    for (unsigned int i = 0; i < n; i++) {
        layers[0]->neurons[i] = input[i];
    }
}

void NeuralNetwork::Forward_Propagation(vec &input) {
    feedInput(input);

    // forward propagation in middle and output layers
    for (unsigned int l = 1; l < layers.size(); l++) {
        layers[l]->Forward_Propagation(layers[l - 1]);
    }
}

void NeuralNetwork::Backward_Propagation(double learning_rate, vec &expected) {
    int n = layers.size();
    Layer *last = layers[n - 1];

    // get error in output layer (init of Backward)
    for (unsigned int i = 0; i < layers[n - 1]->neurons.size(); i++) {
        last->delta[i] = (last->neurons[i] - expected[i]) * (last->neurons[i]) * (1.0 - last->neurons[i]);
    }

    // Backward Propagation in middle layers
    for (unsigned int i = n - 2; i > 0; i--) {
        layers[i]->Backward_Propagation(layers[i + 1]);         // middle layers
    }

    // update weights and bias
    for (unsigned int i = n - 1; i > 0; i--) {
        layers[i]->updateWB(learning_rate, layers[i - 1]);
    }
}

double NeuralNetwork::predict(vec &x) {
    int l = layers.size();
    feedInput(x);
    for (int i = 1; i < l; i++) {
        layers[i]->Forward_Propagation(layers[i - 1]);
    }
    double prediction = round(layers[l - 1]->neurons[0]);
    return prediction;
}


double NeuralNetwork::getError() {
    cout
            << " -------------------------------------------------- ERROR CALCULATION ---------------------------------------------- "
            << endl;
    int n = x.size();
    double error = 0;
    for (int i = 0; i < n; i++) {
        double prediction = predict(x[i]);
        if (fabs(prediction - y[i][0]) > 0.00001) {
            cout << " **** MISS *** ";
            error += 1;
        }
        cout << "Prediction: " << prediction << " Expected: " << y[i][0] << endl;

    }
    return error / n;
}


/* --------------------------------------------------------------------- DEBUG SECTION -------------------------------------------------------- */

void NeuralNetwork::debug(int e) {
    cout
            << "--------------------------------------------------------- SHOW WEIGHTS --------------------------------------------------------- "
            << endl;
    int n = layers.size();
    for ( int l = 0; l < n; l++) {
        cout << "EPOCH : " << e << " LAYER: " << l << endl;
        cout << layers[l]->weights << endl;
        cout << layers[l]->bias << endl;
    }
}

void NeuralNetwork::showGradient(int e) {
    cout
            << "--------------------------------------------------------- SHOW GRADIENT DELTA --------------------------------------------------------- "
            << endl;
    int n = layers.size();
    for (int l = 0; l < n; l++) {
        cout << "EPOCH : " << e << " LAYER: " << l + 1 << endl;
        cout << layers[l]->delta << endl;
    }
}

void NeuralNetwork::showNeurons(int e) {
    cout
            << "--------------------------------------------------------- SHOW NEURONS --------------------------------------------------------- "
            << endl;
    int n = layers.size();
    for ( int l = 0; l < n; l++) {
        cout << "EPOCH " << e << " NEURONS LAYER : " << l << endl;
        cout << layers[l]->neurons << endl;
    }
}



// FALTA utilizar otro vector para calcular errores porque vector acc_error debe ir acumulando el de cada ejemplo del set
// ACTUALIZAR PESOS Y BIAS para cada capa
// http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network
//http://neuralnetworksanddeeplearning.com/chap1.html
// https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4&ab_channel=3Blue1Brown