#include "NeuralNetwork.h"
#include <cstdlib>
#include <cmath>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology) {
    for (int i = 0; i < topology.size(); ++i) {
        layers.emplace_back(Layer(topology[i]));
        if (i > 0) {
            weights.emplace_back(std::vector<std::vector<float>>(topology[i], std::vector<float>(topology[i - 1], 0.0f)));
            initializeWeights(i);
        }
    }
}

void NeuralNetwork::initializeWeights(int layerIndex) {
    for (int i = 0; i < weights[layerIndex - 1].size(); ++i) {
        for (int j = 0; j < weights[layerIndex - 1][i].size(); ++j) {
            weights[layerIndex - 1][i][j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }
}

float NeuralNetwork::sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float NeuralNetwork::sigmoidDerivative(float x) {
    return x * (1.0f - x);
}

void NeuralNetwork::forwardPassCPU(const std::vector<float>& inputs) {
    for (int i = 0; i < inputs.size(); ++i) {
        layers[0].neurons[i].value = inputs[i];
    }

    for (int l = 1; l < layers.size(); ++l) {
        for (int j = 0; j < layers[l].neurons.size(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < layers[l - 1].neurons.size(); ++k) {
                sum += layers[l - 1].neurons[k].value * weights[l - 1][j][k];
            }
            layers[l].neurons[j].value = sigmoid(sum);
        }
    }
}

void NeuralNetwork::backpropagationCPU(const std::vector<float>& expected) {
    // Output layer gradients
    for (int i = 0; i < layers.back().neurons.size(); ++i) {
        float output = layers.back().neurons[i].value;
        layers.back().neurons[i].gradient = (output - expected[i]) * sigmoidDerivative(output);
    }

    // Hidden layer gradients
    for (int l = layers.size() - 2; l > 0; --l) {
        for (int i = 0; i < layers[l].neurons.size(); ++i) {
            float sum = 0.0f;
            for (int j = 0; j < layers[l + 1].neurons.size(); ++j) {
                sum += weights[l][j][i] * layers[l + 1].neurons[j].gradient;
            }
            layers[l].neurons[i].gradient = sum * sigmoidDerivative(layers[l].neurons[i].value);
        }
    }

    // Update weights
    for (int l = 1; l < layers.size(); ++l) {
        for (int i = 0; i < weights[l - 1].size(); ++i) {
            for (int j = 0; j < weights[l - 1][i].size(); ++j) {
                weights[l - 1][i][j] -= 0.1f * layers[l].neurons[i].gradient * layers[l - 1].neurons[j].value;
            }
        }
    }
}

void NeuralNetwork::trainCPU(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expected, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < inputs.size(); ++i) {
            forwardPassCPU(inputs[i]);
            backpropagationCPU(expected[i]);
        }
    }
}
