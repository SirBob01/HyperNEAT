#ifndef HYPER_NEAT_GENOME_H_
#define HYPER_NEAT_GENOME_H_

#include <cassert>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "activations.h"
#include "hyperparams.h"
#include "quadtree.h"
#include "random.h"

namespace HyperNEAT {
    /**
     * The classifications of neural network nodes
     */
    enum NodeType { Input, Hidden, Output };

    /**
     * An edge in the genome network
     */
    struct Edge {
        int from;
        int to;

        bool operator==(const Edge &other) const;
    };

    /**
     * Hash function for edges
     */
    struct EdgeHash {
        std::size_t operator()(Edge const &s) const noexcept;
    };

    /**
     * Information associated with an edge that can be inherited by
     * children during reproduction
     */
    struct EdgeGene {
        double weight;
        bool enabled;
    };

    /**
     * Information associated with a node that can be inherited by
     * children during reproduction
     */
    struct NodeGene {
        double bias;
        activation_t function;

        double activation = 0;
    };

    /**
     * A genome represents the Compositional Pattern-Producing Network
     * (CPPN) used to generate the phenotype Artificial Neural Network
     * (ANN).
     *
     * The ANN is an 2-dimensional substrate, whose points correspond
     * to neurons. The weight between two neurons are determined by feeding
     * their coordinates to the evolved CPPN.
     *
     * To calculate the bias at a point, feed only one point to the CPPN and
     * zero-out the rest, e.g., f(x1, y1, 0, 0).
     */
    class Genome {
        std::unordered_map<Edge, EdgeGene, EdgeHash> _edges;
        std::vector<NodeGene> _nodes;

        std::vector<std::vector<int>> _adjacency;
        std::vector<int> _sorted;

        GenomeParameters _params;

        int _inputs;
        int _outputs;

        double _fitness;

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
         * Add a new node between an existing edge and disable that edge
         */
        void add_node(Edge edge);

        /**
         * Add a new edge between two unconnected nodes, and update
         * the global innovation database
         */
        bool add_edge(Edge edge);

        /**
         * Toggle the enable flag of an edge
         */
        void toggle_enable(Edge edge);

        /**
         * Shift the weight of an edge
         */
        bool shift_weight(Edge edge);

        /**
         * Set the weight of an edge to a new random value
         */
        bool reset_weight(Edge edge);

        /**
         * Shift the bias of a node
         */
        void shift_bias(int node);

        /**
         * Set the bias of a node to a new random value
         */
        void reset_bias(int node);

        /**
         * Change the activation function of a node
         */
        void change_activation(int node);

        /**
         * Test if the node id is an input
         */
        inline bool is_input(int node) { return node < _inputs; }

        /**
         * Test if the node id is an output
         */
        inline bool is_output(int node) {
            return node >= _inputs && node < _inputs + _outputs;
        }

      public:
        Genome(GenomeParameters params);
        Genome(std::vector<NodeGene> &nodes,
               std::unordered_map<Edge, EdgeGene, EdgeHash> &edges,
               GenomeParameters params);
        Genome(const Genome &genome);
        Genome &operator=(const Genome &genome);

        /**
         * Construct a new genome as the offspring of two parents
         */
        Genome(const Genome &a, const Genome &b);

        /**
         * Feed-forward algorithm
         */
        double forward(Point p0, Point p1);

        /**
         * Randomly mutate the genome
         */
        void mutate();

        /**
         * Get the nodes of this network
         */
        const std::vector<NodeGene> &get_nodes() const;

        /**
         * Get the edges of this network
         */
        const std::unordered_map<Edge, EdgeGene, EdgeHash> &get_edges() const;

        /**
         * Calculate the distance to another genome by calculating the
         * weighted sum of differences in their disjoint edges, activations, and
         * weights
         */
        double distance(const Genome &other) const;

        /**
         * Get the fitness score of this genome
         */
        double get_fitness() const;

        /**
         * Set the fitness score of this genome
         */
        void set_fitness(double fitness);
    };
} // namespace HyperNEAT

#endif