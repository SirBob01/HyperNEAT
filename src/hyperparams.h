#ifndef HYPER_NEAT_HYPERPARAMS_H_
#define HYPER_NEAT_HYPERPARAMS_H_

#include "activations.h"

namespace HyperNEAT {
    struct GenomeParameters {
        double m_weight_shift = 0.27;
        double m_weight_change = 0.03;
        double m_bias_shift = 0.27;
        double m_bias_change = 0.03;
        double m_node = 0.01;
        double m_edge = 0.1;
        double m_toggle_enable = 0.1;
        double m_activation = 0.19;

        double mutation_power = 2.5;
    };

    struct PhenomeParameters {
        double variance_threshold = 0.03;
        double division_threshold = 0.1;
        double band_threshold = 0.06;
        double weight_range = 3.0;
        double weight_cutoff = 0.2;
        int initial_depth = 2;
        int maximum_depth = 3;
        int iteration_level = 1;

        // Activation functions of the ANN
        activation_t hidden_activation = lrelu;
        activation_t output_activation = tanh;
    };

    /**
     * Hyperparameters get passed down to individual genomes and phenomes
     */
    struct NEATParameters {
        int population = 150;
        int max_stagnation = 15;
        int max_hall_of_fame = 8;

        double crossover_probability = 0.2;
        double mutation_probability = 0.7;
        double clone_probability = 0.1;

        double c1 = 2.0;
        double c2 = 1.0;

        int target_species = 8;
        double distance_threshold = 1.0;
        double distance_threshold_delta = 0.1;
        double crossover_gene_disable_rate = 0.75;

        double cull_percent = 0.8;

        GenomeParameters genome_params;
        PhenomeParameters phenome_params;
    };
} // namespace HyperNEAT

#endif