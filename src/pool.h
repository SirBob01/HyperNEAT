#ifndef HYPER_NEAT_POOL_H_
#define HYPER_NEAT_POOL_H_

#include <algorithm>
#include <cassert>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <vector>

#include "activations.h"
#include "phenome.h"
#include "species.h"

namespace HyperNEAT {
    /**
     * Container for all genome networks.
     *
     * It evaluates genomes and selects the best ones to repopulate,
     * evolving the pool over time.
     */
    class Pool {
        std::vector<std::unique_ptr<Specie>> _species;

        std::vector<Point> _inputs;
        std::vector<Point> _outputs;
        NEATParameters _params;

        std::vector<std::unique_ptr<Genome>> _elites;
        std::unique_ptr<Genome> _global_best;

        int _generations;

        /**
         * Add a new genome to an existing species, or create a new
         * one if it is too genetically distinct
         */
        void add_genome(std::unique_ptr<Genome> &&genome);

        /**
         * Eliminate the worst in the population
         */
        void cull();

        /**
         * Randomly breed new genomes via mutation or crossover
         */
        void repopulate();

        /**
         * Randomly select a specie whose likelihoods depend on adjusted fitness
         * sum
         */
        Specie &sample_specie();

        /**
         * Read a genome from an input filestream
         */
        std::unique_ptr<Genome> read_genome(std::ifstream &infile);

        /**
         * Write a genome to an output filestream
         */
        void write_genome(std::ofstream &outfile, Genome &genome);

      public:
        Pool(std::vector<Point> inputs,
             std::vector<Point> outputs,
             NEATParameters params);
        Pool(std::string filename, NEATParameters params);

        /**
         * Run an evaluator function through each neural network to calculate
         * fitness and evolve the population
         */
        void evolve(std::function<void(Phenome &phenome)> evaluator);

        /**
         * Get the global fittest neural network
         */
        Phenome get_global_fittest();

        /**
         * Get the current generation number
         */
        int get_generations();

        /**
         * Save the current population to disk
         */
        void save(std::string filename);
    };
} // namespace HyperNEAT

#endif