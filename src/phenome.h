#ifndef HYPER_NEAT_PHENOME_H_
#define HYPER_NEAT_PHENOME_H_

#include <algorithm>
#include <cassert>
#include <queue>
#include <unordered_set>

#include "genome.h"
#include "quadtree.h"

namespace HyperNEAT {
    struct NodePhenotype {
        double bias;
        NodeType type;
        activation_t function;

        double activation = 0;
    };

    /**
     * The neural network to be evaulated. This is defined by the genome CPPN.
     */
    class Phenome {
        Genome &_genome;
        PhenomeParameters _params;
        int _node_count;

        std::vector<Point> &_inputs;
        std::vector<Point> &_outputs;

        // Map 3D substrate points to node indexes
        std::unordered_map<Point, int, PointHash> _pointset;

        std::unordered_map<int, NodePhenotype> _nodes;
        std::unordered_map<Edge, double, EdgeHash> _edges;

        std::unordered_map<int, std::vector<int>> _adjacency;
        std::vector<int> _sorted;

        /**
         * Checks if adding a new edge will create a cycle
         */
        bool creates_cycle(Edge edge);

        /**
         * Query the CPPN (genome) to generate weight and bias values
         */
        double calculate_weight(Point p0, Point p1);

        /**
         * Perform the division and initialization step of the evolving
         * substrate
         */
        Quadtree *division_initialization(Point point, bool outgoing);

        /**
         * Perform the prune and extract algorithm of the evolving substrate
         */
        void
        prune_extract(Point point,
                      Quadtree *quadtree,
                      bool outgoing,
                      std::unordered_map<Edge, double, EdgeHash> &connections);

        /**
         * Check if a path exists between two nodes using BFS
         */
        bool path_exists(int start, int end);

        /**
         * Remove nodes and connections that do not have a path to inputs or
         * outputs
         */
        void cleanup();

        /**
         * Generate the adjacency list
         */
        void generate_adjacency();

        /**
         * Topologically sort the nodes for feed-forward evaluation
         */
        void topological_sort(int node, std::unordered_set<int> &visited);

        /**
         * Update the internal graph structure of the neural network for
         * evaluation
         */
        void update_structure();

        /**
         * Check if all the output nodes are active
         */
        bool active_output();

      public:
        Phenome(Genome &genome,
                std::vector<Point> &inputs,
                std::vector<Point> &outputs,
                PhenomeParameters params);
        Phenome &operator=(const Phenome &rhs);

        /**
         * Evaluate the neural network
         */
        std::vector<double> forward(std::vector<double> input);

        /**
         * Get the fitness of this phenome's genome
         */
        double get_fitness();

        /**
         * Set the fitness of this phenome's genome
         */
        void set_fitness(double fitness);

        /**
         * Get this phenome's genome
         */
        Genome &get_genome();
    };
} // namespace HyperNEAT

#endif