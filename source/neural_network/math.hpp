#ifndef MATH_HPP
#define MATH_HPP

#include <cmath> // exp, log
#include <iostream>

namespace woXrooX{
    namespace math{
        ////////////// Euler's number
        // E ~ 2.718281828459045…
        const float e = 2.718281828459045;

        ////////////// Clip
        float clip(float value, float lower_bound, float upper_bound){
            if(value < lower_bound) return lower_bound;
            else if(value > upper_bound) return upper_bound;
            else return value;
        }

        ////////////// Dot Product
        // Multiplication of arrays
        // arr1[0]*arr2[0] + arr1[1]*arr2[1]..
        float dot_product(float* a, float* b, size_t size){
            float result = 0.0;

            for(size_t i = 0; i < size; i++)
                result += a[i] * b[i];

            return result;

        }

        ////////////// ReLU (Rectified Linear Unit) (Activation Function)
        // y = x if x > 0 else 0
        float ReLU(float value){
            return value > 0 ? value : 0;
        }

        ////////////// Softmax (Activation Function)
        // softmax(xᵢ) = e^(xᵢ) / (∑(e^(xⱼ)))
        // Input -> Exponentiate -> Normalize - > Output

        //// Exponential Activation
        // y = E^x, E ~ 2.718281828459045…
        void exponentiate(float* values, size_t size){
            for(size_t i = 0; i < size; i++)
                values[i] = exp(values[i]);
        }

        //// Normalization
        // X1 = X1 / ( X1 + X2 + Xn)
        void normalize(float* values, size_t size){
            float sum = 0;

            for(size_t i = 0; i < size; i++)
                sum += values[i];

            for(size_t i = 0; i < size; i++)
                values[i] = values[i] / sum;

        }


        ////////////// Categorical Cross Entropy (AKA Softmax Loss)
        // Example Of One-Hot Encoding:
        // outputs = [0.7, 0.2, 0.1]
        // target_outputs = [1, 0, 0]
        // loss += target_outputs[i] * log(outputs[i])
        // loss = loss * -1
        float CCE(float* values, int* target_outputs, size_t size){
            float loss = 0.0;
            for(size_t i = 0; i < size; i++)
                loss += target_outputs[i] * std::log(values[i]);

            loss = -loss;

            return loss;
        }

        // Simplified Version Of CCE
        // In CCE: Most of the target_outputs values are zero.
        // 0 * log(something) = 0 so we are just working on where target_outputs[x] = 1
        // And end formula becomes = -(1 * log(outputs[x]))
        float loss(float value){
            return -std::log(
                clip(value, 0.0000001, 0.9999999)
            );

        }

        ////////////// Max
        float max(float* values, size_t size){
            float biggest_value = values[0];

            for(size_t i = 0; i < size; i++)
                if(values[i] > biggest_value) biggest_value = values[i];

            return biggest_value;
        }
    }
}

#endif
