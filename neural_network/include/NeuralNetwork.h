#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "Layer.h"

class NeuralNetwork {
public:
    std::vector<Layer> layers;
    std::vector<std::vector<std::vector<float>>> weights;

    NeuralNetwork(const std::vector<int>& topology);
    void forwardPassCPU(const std::vector<float>& inputs);
    void backpropagationCPU(const std::vector<float>& expected);
    void trainCPU(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expected, int epochs);

private:
    void initializeWeights(int layerIndex);
    float sigmoid(float x);
    float sigmoidDerivative(float x);
};

#endif // NEURAL_NETWORK_H
