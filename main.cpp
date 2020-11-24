#include <iostream>
#include "Dataset.h"
#include "NeuralNetwork.h"
#include "time.h"
#include "iostream"
#include <chrono>

using namespace std::chrono;


int main(int n_args, char** args) {
    srand(time(NULL)); //seed rand

    int n_layers = 0;
    int n_neurons = 0;
    double tol = 0.1;

    n_layers = atoi(args[2]);
    n_neurons = atoi(args[3]);
    if (n_args > 4)
        tol = atof(args[4]);

    if (n_layers <= 0 || n_neurons <= 0 || tol <= 0.0 ||n_args < 4){
        std::cout << "Parametros incorrectos " << endl;
        return 1;
    }

    // read Dataset
    Dataset dataset(args[1]);
    dataset.Normalize();

    // build neural network
    NeuralNetwork N(n_layers,n_neurons, dataset);

    //clock start
    high_resolution_clock::time_point start = high_resolution_clock::now();

    N.startTrainning(100, 0.5, tol);

    //calculo de tiempo
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> timeElapsed = duration_cast<duration<double>>(end-start);
    cout << "TIEMPO DE ENTRENAMIENTO: " ;
    cout << timeElapsed.count() << " seconds " << endl;

    //N.~NeuralNetwork();
    return 0;
}
