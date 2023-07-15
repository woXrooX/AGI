#include <iostream>

#include "helpers.hpp"
#include "layer.hpp"

int main(int argc, char const *argv[]){

    const size_t INPUTS_SIZE = 4;
    const size_t OUTPUT_NEURONS_SIZE = 3;

    float* X = new float[INPUTS_SIZE];
    X[0] = 4.8;
    X[1] = 1.21;
    X[2] = 2.385;
    X[3] = 0;

    /////// Input Layer
    woXrooX::Layer input_layer(INPUTS_SIZE, OUTPUT_NEURONS_SIZE, X);

    /////// Hidden Layers
    woXrooX::Layer hidden_layer_1(5);
    hidden_layer_1.forward(&input_layer);

    woXrooX::Layer hidden_layer_2(5);
    hidden_layer_2.forward(&hidden_layer_1);

    woXrooX::Layer hidden_layer_3(5);
    hidden_layer_3.forward(&hidden_layer_2);

    /////// Output Layer
    woXrooX::Layer output_layer(2);
    output_layer.forward(&hidden_layer_3);

    // Softmax
    woXrooX::math::exponentiate(output_layer.getOutputs(), output_layer.getOutputsSize());
    woXrooX::math::normalize(output_layer.getOutputs(), output_layer.getOutputsSize());
    output_layer.calcLoss();
    output_layer.info();

    return 0;

}
