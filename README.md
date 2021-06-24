# HyperNEAT

C++ ES-HyperNEAT algorithm implementation

## Algorithm

The HyperNEAT algorithm uses evolution to train neural networks. Genome neural networks, known as Compositional Pattern-Producing Networks (CPPN), are randomly generated via mutations and crossover. This is then used to "paint" a pattern on a 4-dimensional hypercube, which represents the weights and biases of the resulting phenome neural network.

Through the process of Darwinian natural selection, the genomes will eventually converge towards creating a network that can maximize the fitness function and solve the task.

## License

Code and documentation Copyright (c) 2021 Keith Leonardo

Code released under the [MIT License](https://choosealicense.com/licenses/mit/).