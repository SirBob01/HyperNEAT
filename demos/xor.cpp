#include <HyperNEAT.h>
#include <vector>

struct Case {
    std::vector<double> input;
    std::vector<double> output;
};

int main() {
    HyperNEAT::randseed();

    const int max_generations = 200;
    const std::vector<Case> cases = {
        {{0, 0}, {0}},
        {{1, 1}, {0}},
        {{1, 0}, {1}},
        {{0, 1}, {1}},
    };
    const std::vector<HyperNEAT::Point> input = {{-0.5, -0.5}, {0.5, -0.5}};
    const std::vector<HyperNEAT::Point> output = {{0, 0.5}};

    HyperNEAT::NEATParameters params;
    params.phenome_params.output_activation = HyperNEAT::sigmoid;
    params.phenome_params.division_threshold = 0.2;
    params.phenome_params.band_threshold = 0.1;
    params.phenome_params.initial_depth = 1;
    params.phenome_params.maximum_depth = 2;
    HyperNEAT::Pool pool(input, output, params);

    for (int i = 0; i < max_generations; i++) {
        std::cout << "Generation " << pool.get_generations() << "\n";

        // Calculate the fitness of each network
        auto phenomes = pool.get_phenomes();
        for (auto phenome : phenomes) {
            double total_error = 0.0;
            for (auto &c : cases) {
                double result = phenome->forward(c.input)[0];
                double diff = c.output[0] - result;
                total_error += diff * diff;
            }
            phenome->set_fitness(1 / (1 + std::sqrt(total_error)));
        }

        // Print the result of the fittest phenome
        auto phenome = pool.get_fittest();
        double total_error = 0.0;
        for (auto &c : cases) {
            double result = phenome.forward(c.input)[0];
            double diff = c.output[0] - result;
            total_error += diff * diff;
            std::cout << c.input[0] << " ^ " << c.input[1] << " = " << result
                      << "\n";
        }
        std::cout << "Accuracy " << 1 / (1 + std::sqrt(total_error)) << "\n\n";

        // Evolve the pool
        pool.evolve();
    }
    return 0;
}