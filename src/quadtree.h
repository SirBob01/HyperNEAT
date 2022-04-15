#ifndef HYPER_NEAT_QUADTREE_H_
#define HYPER_NEAT_QUADTREE_H_

#include <vector>

namespace HyperNEAT {
    /**
     * A 2-dimensional point on the substrate
     */
    struct Point {
        double x;
        double y;

        bool operator==(const Point &other) const;
    };

    /**
     * Hash function for points
     */
    struct PointHash {
        std::size_t operator()(Point const &s) const noexcept;
    };

    /**
     * Recursively divides a space into quadrants
     */
    struct Quadtree {
        Point center;
        double size;
        int level;

        double weight;

        std::vector<Quadtree *> children;

        Quadtree(Point center, double size, int level) :
            center(center), size(size), level(level){};
        ~Quadtree();

        /**
         * Generate this node's children
         */
        void generate_children();

        /**
         * Calculate the variance of the Quadtree
         */
        double get_variance();

      private:
        /**
         * Recursively grab the weights of all leaf nodes
         */
        void recur_weights(Quadtree *root, std::vector<double> &weights);
    };

    double variance(std::vector<double> &values);
} // namespace HyperNEAT

#endif