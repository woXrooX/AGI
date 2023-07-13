#ifndef X_AI_HPP
#define X_AI_HPP

#include "helpers.hpp"
#include "layer.hpp"
#include <iostream>

namespace woXrooX{

    void init(){
        const size_t INPUTS_SIZE = 4;
        const size_t OUTPUT_NEURONS_SIZE = 3;
        float* X = new float[INPUTS_SIZE];
        // X[0] = 1.0;
        // X[1] = 2.0;
        // X[2] = 3.0;
        // X[3] = 2.5;

        X[0] = 4.8;
        X[1] = 1.21;
        X[2] = 2.385;
        X[3] = 0;

        /////// Input Layer
        Layer input_layer(INPUTS_SIZE, OUTPUT_NEURONS_SIZE, X);

        /////// Hidden Layers
        Layer hidden_layer_1(5);
        hidden_layer_1.forward(&input_layer);

        Layer hidden_layer_2(5);
        hidden_layer_2.forward(&hidden_layer_1);

        Layer hidden_layer_3(5);
        hidden_layer_3.forward(&hidden_layer_2);

        /////// Output Layer
        Layer output_layer(2);
        output_layer.forward(&hidden_layer_3);

        // Softmax
        math::exponentiate(output_layer.getOutputs(), output_layer.getOutputsSize());
        math::normalize(output_layer.getOutputs(), output_layer.getOutputsSize());
        output_layer.calcLoss();
        output_layer.info();

    }


}

#endif
