#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neuron.h"

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int size) {
        neurons.resize(size);
    }
};

#endif // LAYER_H
