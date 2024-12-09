#include <iostream>
#include <chrono>
#include "NeuralNetwork.h"
#include "NeuralNetworkGPU.h"

int main() {
    NeuralNetwork nn({2, 3, 1});

    std::vector<std::vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<float>> outputs = {{0}, {1}, {1}, {0}};

    // CPU Training
    auto startCPU = std::chrono::high_resolution_clock::now();
    nn.trainCPU(inputs, outputs, 1000);
    auto endCPU = std::chrono::high_resolution_clock::now();

    std::cout << "Tiempo en CPU: " << std::chrono::duration<float>(endCPU - startCPU).count() << " segundos\n";

    // GPU Training
    auto startGPU = std::chrono::high_resolution_clock::now();
    trainOnGPU(inputs, outputs, 2, 3, 1, 1000);
    auto endGPU = std::chrono::high_resolution_clock::now();

    std::cout << "Tiempo en GPU: " << std::chrono::duration<float>(endGPU - startGPU).count() << " segundos\n";

    return 0;
}
