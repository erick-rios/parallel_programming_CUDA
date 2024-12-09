#ifndef NEURAL_NETWORK_GPU_H
#define NEURAL_NETWORK_GPU_H

#include <vector>

// Función para entrenar en GPU
void trainOnGPU(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& outputs, 
                int inputSize, int outputSize, int hiddenSize, int epochs);

#endif
