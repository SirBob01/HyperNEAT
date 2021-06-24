#include "random.h"

namespace HyperNEAT {
    double random() {
        return uniform(rng);
    }

    void randseed() {
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    int randrange(int start, int stop) {
        return random() * (stop - start) + start;
    }
}