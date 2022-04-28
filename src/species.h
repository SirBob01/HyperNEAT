#ifndef HYPER_NEAT_SPECIES_H_
#define HYPER_NEAT_SPECIES_H_

#include <algorithm>
#include <memory>
#include <queue>

#include "genome.h"
#include "hyperparams.h"

namespace HyperNEAT {
    /**
     * A container for genomes of similar topology and weights.
     * Species are evaluated independently to protect variation in the gene
     * pool
     */
    class Specie {
        std::vector<std::unique_ptr<Genome>> _members;
        NEATParameters _params;
        int _stagnation_count;

        double _fitness_sum;
        std::queue<double> _fitness_history;

      public:
        // Network parameters are going to propagated to all genomes
        Specie(std::unique_ptr<Genome> &&representative, NEATParameters params);
        Specie(const Specie &other);

        /**
         * Add a new genome to the specie
         */
        void add(std::unique_ptr<Genome> &&genome);

        /**
         * Get the representative genome of the specie for distance matching
         */
        Genome &get_repr();

        /**
         * Get the number of members in this specie
         */
        int get_size();

        /**
         * Get the members of the species
         */
        const std::vector<std::unique_ptr<Genome>> &get_members();

        /**
         * Update the total average fitness of the specie for the current
         * generation
         *
         * This will be used to determine whether to kill this specie
         */
        void update_fitness();

        /**
         * Return the adjusted fitness sum
         */
        double get_fitness_sum();

        /**
         * Get a random genome from this specie.
         *
         * Genomes with a higher fitness score are more likely to be selected
         */
        Genome &sample();

        /**
         * Get the best genome in the species
         */
        Genome &get_best();

        /**
         * Cull the worst performing of the species
         */
        void cull();

        /**
         * Check if the specie will be allowed to survive
         */
        bool can_progress();
    };
} // namespace HyperNEAT

#endif