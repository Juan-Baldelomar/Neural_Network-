#include <iostream>
#include "Dataset.h"
#include "NeuralNetwork.h"
#include "time.h"
#include "iostream"
#include "set"

using namespace std;

int main() {
    srand(time(NULL));
    Dataset dataset("Input/data.dat");
    dataset.Normalize();
    NeuralNetwork N(1,2, dataset);
    N.startTrainning(100, 0.5);
    vec x(4, 0);

    for (int i = 0; i<dataset.x.size(); i++){
        x[0] = dataset.x[i][0];
        x[1] = dataset.x[i][1];
        x[2] = dataset.x[i][2];
        x[3] = dataset.x[i][3];
        N.predict(x);
    }
    return 0;
}
