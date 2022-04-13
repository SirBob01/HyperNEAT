#include "activations.h"

namespace HyperNEAT {
    std::unordered_map<std::string, activation_t> activations = {
        {"abs", std::fabs},
        {"tanh", tanh},
        {"sigmoid", sigmoid},
        {"gaussian", gaussian},
        {"sin", std::sin},
        {"identity", identity},
        {"lrelu", lrelu},
    };
} // namespace HyperNEAT