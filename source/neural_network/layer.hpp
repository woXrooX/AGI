#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <random>
#include <ctime>

#include "math.hpp"

namespace woXrooX{
    class Layer{
    public:
        // Input Layer
        Layer(size_t input_size, size_t output_size, float* X) :
            input_size(input_size),
            output_size(output_size),
            inputs(X)
        {
            this->id = ++Layer::ID;

            // Seed fir randomizing
            srand(time(NULL));

            this->genWeights();
            this->genBiases();

            this->calcOutputs();

        }

        // Empty Init
        // Hidden Layer
        Layer(size_t output_size) :
            output_size(output_size)
        {
            this->id = ++Layer::ID;

            srand(time(NULL));

        }

        ~Layer(){
            for (size_t i = 0; i < this->output_size; i++)
                delete[] this->weights[i];

            delete[] this->weights;

            delete[] this->biases;

        }

        //////////////// APIs
        void forward(Layer* input_layer){
            this->input_size = input_layer->getOutputsSize();
            this->inputs = input_layer->getOutputs();

            this->genWeights();
            this->genBiases();

            this->calcOutputs();

        }

        //////////////// Helpers
        void info(){
            std::cout << "==================================== Layer ID: " << this->id << " ====================================" << "\n";
            std::cout << "Input size: " << this->input_size << '\n';
            std::cout << "Output size: " << this->output_size << '\n';

            // Inputs
            std::cout << "\n-------- Inputs --------" << "\n";
            for(size_t i = 0; i < this->input_size; i++)
                std::cout << this->inputs[i] << "\t\t";

            // Weights
            std::cout << "\n-------- Weights --------" << "\n";
            for (size_t i = 0; i < this->output_size; i++) {
                for (size_t j = 0; j < this->input_size; j++) {
                    std::cout << this->weights[i][j] << "\t";
                }
                std::cout << "\n";
            }

            // Biases
            std::cout << "-------- Biases --------" << "\n";
            for(size_t i = 0; i < this->output_size; i++)
                std::cout << this->biases[i] << "\t";

            // Outpus
            std::cout << "\n-------- Outputs --------" << "\n";
            for(size_t i = 0; i < this->output_size; i++)
                std::cout << this->outputs[i] << "\t";
            std::cout << '\n';

            // Loss
            std::cout << "\n-------- Loss --------" << "\n";
            std::cout << this->loss << '\n';

            std::cout << "\n=====================================================================================" << "\n";
        }

        //////////////// Getters
        const size_t getOutputsSize() const {
            return this->output_size;
        }

        float* getOutputs() const {
            return this->outputs;
        }

        void calcLoss(){
            this->loss = math::loss(
                math::max(this->outputs, this->output_size)
            );
        }

        //////////////// Setters


    private:
        void genWeights(){
            this->weights = new float*[this->output_size];

            for(size_t i = 0; i < this->output_size; i++){

                this->weights[i] = new float[this->input_size];

                for(size_t j = 0; j < this->input_size; j++)
                    this->weights[i][j] = this->random() * 0.10;

            }

        }

        void genBiases(){
            this->biases = new float[this->output_size];

            for(size_t i = 0; i < this->output_size; i++)
                this->biases[i] = this->random();

        }

        //////////////// Math
        ////// Retursn float -1.0 to 1.0
        float random(){
            float max = 1.0;
            float min = -1.0;

            return ((float(rand()) / float(RAND_MAX)) * (max - min)) + min;;
        }

        void calcOutputs(){
            this->outputs = new float[this->output_size];

            for(size_t i = 0; i < this->output_size; i++)
                this->outputs[i] =
                    math::ReLU(
                        math::dot_product(this->inputs, this->weights[i], this->input_size) +
                        this->biases[i]
                    );
        }

        //////////////// Variables
        unsigned int id = 0;
        size_t input_size = 0;
        size_t output_size = 0;
        float loss;
        float* inputs;
        float** weights;
        float* biases;
        float* outputs;

        static unsigned int ID;

    };

    unsigned int Layer::ID = 0;

}

#endif
