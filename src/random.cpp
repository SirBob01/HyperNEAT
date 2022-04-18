#include "random.h"

namespace HyperNEAT {
    double random() { return uniform(rng); }

    void randseed() {
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    int randrange(int start, int stop) {
        return random() * (stop - start) + start;
    }

    void probability_function(FunctionalDistribution &distribution) {
        double r = random();
        double cumulative = 0;
        for (auto &pair : distribution) {
            if (r > cumulative && r <= cumulative + pair.first) {
                pair.second();
                return;
            }
            cumulative += pair.first;
        }
    }
} // namespace HyperNEAT