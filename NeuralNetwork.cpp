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

    // All Data
    x.assign(n_training, vec(n_init, 0));
    y.assign(n_training, 0);
    y_feat.assign(n_training, vec(2, 0));

    // Mini Batch data
    Mx.assign(n_training/10, vec(n_init, 0));
    My.assign(n_training/10, 0);
    My_feat.assign(n_training/10, vec(2, 0));

    //get data from dataset
    for (int i = 0; i<n_training; i++){
        y[i] = dataset.y[i];
        if (y[i] == 1){
            y_feat[i][0] = 1;
        }else{
            y_feat[i][1] = 1;
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
    Input = new Layer(-1, n_neurons, n_init);
    Output = new Layer(n_neurons, -1, n_exit);
    layers.assign(n_layers, 0);
    for (int l = 0; l < n_layers; l++){
        if (l == 0 && l!= n_layers-1)                                                       // fist middle layer and more than one middle layer
            layers[0] = new Layer(n_init, n_neurons, n_neurons);
        else if (l == n_layers - 1 && l!= 0)                                                // last middle layer and more than one middle layer
            layers[l] = new Layer(n_neurons, n_exit, n_neurons);
        else if (l == 0)                                                                    // one middle layer
            layers[l] = new Layer(n_init, n_exit, n_neurons);
        else                                                                                // middle layers
            layers[l] = new Layer(n_neurons, n_neurons, n_neurons);
    }
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
        buildMiniBatch();
        Forward_Propagation(learning_rate);

        Output->updateWB(learning_rate/Mx.size());
        for (int i = n-1; i>=0; i--){
            layers[i]->updateWB(learning_rate/Mx.size());
        }
        double error = getError();
        if ( error < 0.5)
            break;
        cout << "error: " << error << endl;
    }
}

void NeuralNetwork::debug(int e) {
    cout << "--------------------------------------------------------- SHOW WEIGHTS --------------------------------------------------------- " << endl;
    int n = layers.size();
    cout << "EPOCH : " << e << " LAYER: INPUT" << endl;
    cout << Input->weights << endl;
    for (int l = 0; l< n; l++){
        cout << "EPOCH : " << e << " LAYER: " << l+1 << endl;
        cout << layers[l]->weights << endl;
    }
    cout << "EPOCH : " << e << " LAYER: OUTPUT" << endl;
    cout << Output->weights << endl;

}

void NeuralNetwork::showGradient(int e) {
    cout << "--------------------------------------------------------- SHOW GRADIENT --------------------------------------------------------- " << endl;
    int n = layers.size();
    for (int l = 0; l< n; l++){
        cout << "EPOCH : " << e << " LAYER: " << l+1 << endl;
        cout << layers[l]->weights_grad << endl;
        cout << layers[l]->bias_grad << endl;
    }
    cout << "EPOCH : " << e << " LAYER: OUTPUT" << endl;
    cout << " --- WEIGHTS GRAD: ---- " << endl;
    cout << Output->weights_grad << endl;
    cout << " --- BIAS GRAD: ---- " << endl;
    cout << Output->bias_grad << endl;

}

void NeuralNetwork::showNeurons(int e) {
    cout << "--------------------------------------------------------- SHOW NEURONS --------------------------------------------------------- " << endl;

    int n = layers.size();
    cout << "NEURONS LAYER : INPUT" << endl;
    cout << Input->neurons<< endl;
    for (int l = 0; l<n; l++){
        cout << "EPOCH " << e << " NEURONS LAYER : " << l << endl;
        cout << layers[l]->neurons<< endl;
    }
    cout << "NEURONS LAYER : OUTPUT" << endl;
    cout << Output->neurons<< endl;
}

void NeuralNetwork::cleanDeltas() {
    int l = layers.size();

    CleanVector(Output->bias_grad);
    CleanMatrix(Output->weights_grad);

    for (int k = 0; k<l; k++){
        CleanMatrix(layers[k]->weights_grad);
        CleanVector(layers[k]->bias_grad);
    }
}

void NeuralNetwork::Forward_Propagation(double learning_rate) {
    int n = layers.size();
    cleanDeltas();

    // for every x in the trainning sample
    for (int i = 0; i<Mx.size(); i++){
        for (int j = 0; j< Mx[0].size(); j++){
            Input->neurons[j] = Mx[i][j];
        }
        // forward propagation
        for (int l = 0; l<layers.size(); l++){
            if (l==0){
                layers[0]->Forward_Propagation(Input);
            } else{
                layers[l]->Forward_Propagation(layers[l-1]);
            }
        }
        Output->Forward_Propagation(layers[n-1]);
        //showNeurons(0);

        Backward_Propagation(learning_rate, i);
        //showGradient(0);
    }
}

void NeuralNetwork::Backward_Propagation(double  learning_rate, int sample_i) {
    int n = layers.size();
    vector<double> gradient(Output->neurons.size(), 0), error(Output->z.size(), 0);

    for (int i = 0; i<error.size(); i++)
        error[i] = sigmoid_prime(Output->z[i]);                            // sig'(z)

    Resta_Vector(Output->neurons, My_feat[sample_i], gradient);      // gradient = neurons - y_real
    Hadamard_Product(gradient, error, Output->delta);               // acc_error =  (neurons - y_real) * (sig'(z))

    // Gradient accumulation
    Output->addWeightsGrad(layers[n-1]->neurons);

    // Backward Propagation
    for (int i = n-1; i>=0; i--){
        if (i == n-1 && i!= 0)
            layers[i]->Backward_Propagation(Output->weights, Output->delta, layers[i-1]->neurons);                   // last middle layer and more than 1 middle layer
        else if (i==0 && i!= n-1)
            layers[i]->Backward_Propagation(layers[i+1]->weights, layers[i+1]->delta, Input->neurons);               // first middle layer and more than 1 middle layer
        else if (i == 0)
            layers[0]->Backward_Propagation(Output->weights, Output->delta, Input->neurons);                         // just 1 middle layer
        else
            layers[i]->Backward_Propagation(layers[i+1]->weights, layers[i+1]->delta, layers[i-1]->neurons);         // middle layers
    }
}

double NeuralNetwork::predict(vec &x) {
    int n = x.size();
    int l = layers.size();
    //debug(-1);
    for (int i = 0; i<n; i++){
        Input->neurons[i] = x[i];
    }

    for (int i = 0; i<l; i++){
        if (i==0)
            layers[0]->Forward_Propagation(Input);
        else
            layers[i]->Forward_Propagation(layers[i-1]);
    }

    Output->Forward_Propagation(layers[l-1]);

    double max = 0;
    int i_max = 0;
    for (int i = 0; i<Output->neurons.size(); i++){
        if (max < Output->neurons[i]){
            max = Output->neurons[i];
            i_max = i;
        }
    }
    double prediction = 0;
    if (i_max == 0)
        prediction = 1;

    cout << "Prediction :" << prediction << endl;
    return  prediction;
}


double NeuralNetwork::getError() {
    cout << " -------------------------------------------------- OUTPUT NEURONS ---------------------------------------------- " << endl;
    int n = x.size();
    double error = 0;
    for (int i = 0; i<n; i++){
        double prediction = predict(x[i]);
        if (prediction != y[i])
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