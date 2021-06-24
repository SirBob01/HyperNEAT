#ifndef HYPER_NEAT_ACTIVATIONS_H_
#define HYPER_NEAT_ACTIVATIONS_H_

#include <string>
#include <unordered_map>
#include <cmath>

namespace HyperNEAT {
    using activation_t = double (*)(double x);

    inline double lrelu(double x) {
        double slope = 0.05;
        return std::max(x, slope * x);
    }

    inline double sigmoid(double x) {
        return 1.0/(1.0+std::exp(-x));
    }

    inline double tanh(double x) {
        return 2.0 * sigmoid(2 * x) - 1.0;
    }

    inline double gaussian(double x) {
        return std::exp(-0.5 * x * x);
    }

    // Taken from https://arxiv.org/abs/2006.08195
    inline double zihaue_periodic(double x) {
        double z = std::sin(x);
        return x + z * z;
    }

    inline double identity(double x) {
        return x;
    }

    const std::unordered_map<std::string, activation_t> activations = {
        {"abs", std::fabs},
        {"tanh", tanh},
        {"sigmoid", sigmoid},
        {"gaussian", gaussian},
        {"sin", std::sin},
        {"identity", identity},
        {"lrelu", lrelu},
    };
}

#endif