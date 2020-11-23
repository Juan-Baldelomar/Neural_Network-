//
// Created by juan on 20/11/20.
//

#include "Dataset.h"
#include "sstream"
#include "iostream"
#include "fstream"
#include "set"

// UTILITIES
int getRandom(int base, int limit) {
    return base + rand() % (limit - base + 1);
}

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

    set<int> elements;
    int  i = 0;

    //shuffle data
    while (i != y.size()){
        int pos = getRandom(0, y.size()-1);         //get random position in range(0, y.size)
        auto exists = elements.find(pos);                     //find if element exists in elements
        if (exists != elements.end())                         //if exists, next iteration
            continue;

        // if element does not exist, add data to x and y
        elements.insert(pos);
        for (int j = 0; j<m-1; j++){
            x[i][j] = data[pos][j];
        }
        y[i] = data[pos][m-1];
        i++;
    }
}

void Dataset::Normalize() {
    int n = x.size();
    int m = x[0].size();
    for (int j = 0; j<m; j++){
        double max = x[0][j];
        double min = x[0][j];
        for (int i = 0; i<n; i++){
            if (max < x[i][j])
                max = x[i][j];
            if (min > x[i][j])
                min = x[i][j];
        }

        for (int i = 0; i<n; i++){
            x[i][j] = (x[i][j] - min)/(max-min);
        }
    }
}