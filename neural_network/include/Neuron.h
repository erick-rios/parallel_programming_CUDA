#ifndef NEURON_H
#define NEURON_H

class Neuron {
public:
    float value;
    float gradient;

    Neuron() : value(0.0f), gradient(0.0f) {}
};

#endif // NEURON_H
