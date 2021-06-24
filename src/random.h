#ifndef HYPER_NEAT_RANDOM_H_
#define HYPER_NEAT_RANDOM_H_

#include <random>
#include <chrono>

namespace HyperNEAT {
    static std::default_random_engine rng;
    static std::uniform_real_distribution<double> uniform(0, 1);

    double random();

    void randseed();

    int randrange(int start, int stop);
}

#endif