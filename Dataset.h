//
// Created by juan on 20/11/20.
//

#ifndef NEURAL_NETWORK_DATASET_H
#define NEURAL_NETWORK_DATASET_H
#include "string"
#include "vector"
using  namespace std;

class Dataset {
public:
    vector<vector<double>>x;
    vector<double>y;
public:
    Dataset(string name);
};


#endif //NEURAL_NETWORK_DATASET_H
