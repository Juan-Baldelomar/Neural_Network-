#include <iostream>
#include "Dataset.h"
#include "NeuralNetwork.h"
#include "time.h"
#include "iostream"
#include <chrono>

using namespace std::chrono;


int main() {
    srand(time(NULL)); //seed rand

    // read Dataset
    Dataset dataset("Input/data.dat");
    dataset.Normalize();

    // build neural network
    NeuralNetwork N(1,3, dataset);

    //clock start
    high_resolution_clock::time_point start = high_resolution_clock::now();

    N.startTrainning(500, 0.5);

    //calculo de tiempo
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> timeElapsed = duration_cast<duration<double>>(end-start);
    cout << "TIEMPO DE ENTRENAMIENTO: " ;
    cout << timeElapsed.count() << " seconds " << endl;

    /*vec x(4, 0);
    for (int i = 0; i<dataset.x.size(); i++){
        x[0] = dataset.x[i][0];
        x[1] = dataset.x[i][1];
        x[2] = dataset.x[i][2];
        x[3] = dataset.x[i][3];
        N.predict(x);
    }*/
    return 0;
}
