//
// Created by juan on 20/11/20.
//

#ifndef NEURAL_NETWORK_DATASET_H
#define NEURAL_NETWORK_DATASET_H
#include "string"
#include "vector"
using  namespace std;

int getRandom(int base, int limit);

class Dataset {
public:
    vector<vector<double>>x;
    vector<double>y;
public:
    Dataset(string name);
    void Normalize();
};


#endif //NEURAL_NETWORK_DATASET_H
