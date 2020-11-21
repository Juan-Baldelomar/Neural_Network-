//
// Created by juan on 20/11/20.
//

#include "Dataset.h"
#include "sstream"
#include "iostream"
#include "fstream"

Dataset::Dataset(string filename) {
    ifstream file(filename);
    stringstream caster;
    string line;
    double number;
    vector<vector<double>> data;
    getline(file, line);                                 //read header
    while(getline(file, line)){
        data.push_back(vector<double>());
        int pos = data.size()-1;
        stringstream ss(line);
        string splitLine;
        while(getline(ss, splitLine, '\t')){
            caster.clear();
            caster << splitLine;
            caster >> number;
            data[pos].push_back(number);
        }
    }
    int n = data.size();
    int m = data[0].size();
    x.assign(n, vector<double>(m-1, 0));
    y.assign(n, 0);

    for (int i = 0; i<n; i++){
        for (int j = 0; j<m-1; j++){
            x[i][j] = data[i][j];
        }
        y[i] = data[i][m-1];
    }
}