#include "NeuralNetworkGPU.h"
#include <cuda_runtime.h>
#include <iostream>

// Kernel para la propagación hacia adelante
__global__ void forwardPassKernel(float* inputs, float* weights, float* outputs, int inputSize, int outputSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; ++i) {
            sum += inputs[i] * weights[i * outputSize + idx];
        }
        outputs[idx] = 1.0f / (1.0f + expf(-sum)); // Sigmoid
    }
}

// Kernel para calcular gradientes de salida
__global__ void computeOutputGradientsKernel(float* outputs, float* targets, float* gradients, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        gradients[idx] = (outputs[idx] - targets[idx]) * outputs[idx] * (1.0f - outputs[idx]);
    }
}

// Kernel para actualizar los pesos
__global__ void updateWeightsKernel(float* weights, float* inputs, float* gradients, int inputSize, int outputSize, float learningRate) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < inputSize * outputSize) {
        int i = idx / outputSize;
        int j = idx % outputSize;
        weights[idx] -= learningRate * inputs[i] * gradients[j];
    }
}

// Función para entrenar en GPU
void trainOnGPU(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& outputs, 
                int inputSize, int outputSize, int hiddenSize, int epochs) {
    // Allocate memory on the GPU
    float *d_inputs, *d_weights, *d_outputs, *d_targets, *d_gradients;
    cudaMalloc(&d_inputs, inputSize * sizeof(float));
    cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
    cudaMalloc(&d_outputs, outputSize * sizeof(float));
    cudaMalloc(&d_targets, outputSize * sizeof(float));
    cudaMalloc(&d_gradients, outputSize * sizeof(float));

    // Initialize weights (for simplicity, using random values)
    std::vector<float> weights(inputSize * outputSize);
    for (int i = 0; i < inputSize * outputSize; ++i) {
        weights[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy weights to GPU
    cudaMemcpy(d_weights, weights.data(), inputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Copy inputs and targets to GPU
            cudaMemcpy(d_inputs, inputs[i].data(), inputSize * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_targets, outputs[i].data(), outputSize * sizeof(float), cudaMemcpyHostToDevice);

            // Launch forward pass kernel
            forwardPassKernel<<<(outputSize + 255) / 256, 256>>>(d_inputs, d_weights, d_outputs, inputSize, outputSize);
            cudaDeviceSynchronize();

            // Launch compute gradients kernel
            computeOutputGradientsKernel<<<(outputSize + 255) / 256, 256>>>(d_outputs, d_targets, d_gradients, outputSize);
            cudaDeviceSynchronize();

            // Launch update weights kernel
            updateWeightsKernel<<<(inputSize * outputSize + 255) / 256, 256>>>(d_weights, d_inputs, d_gradients, inputSize, outputSize, 0.01f);
            cudaDeviceSynchronize();
        }
    }

    // Free GPU memory
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_outputs);
    cudaFree(d_targets);
    cudaFree(d_gradients);
}