#ifndef HYPER_NEAT_RANDOM_H_
#define HYPER_NEAT_RANDOM_H_

#include <chrono>
#include <random>

namespace HyperNEAT {
    static std::default_random_engine rng;
    static std::uniform_real_distribution<double> uniform(0, 1);

    /**
     * Generate a random uniform double (0-1)
     */
    double random();

    /**
     * Seed the RNG
     */
    void randseed();

    /**
     * Generate a random integer within the given range
     */
    int randrange(int start, int stop);
} // namespace HyperNEAT

#endif