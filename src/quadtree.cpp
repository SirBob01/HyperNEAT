#include "quadtree.h"

namespace HyperNEAT {
    bool Point::operator==(const Point &other) const {
        return other.x == x && other.y == y;
    }

    std::size_t PointHash::operator()(Point const &s) const noexcept {
        std::size_t h1 = std::hash<int>{}(s.x);
        std::size_t h2 = std::hash<int>{}(s.y);
        return h1 ^ (h2 << 1);
    }

    void Quadtree::generate_children() {
        Point quadrant_centers[4] = {
            {center.x - size / 2, center.y - size / 2},
            {center.x + size / 2, center.y - size / 2},
            {center.x - size / 2, center.y + size / 2},
            {center.x + size / 2, center.y + size / 2},
        };
        for (auto &center : quadrant_centers) {
            children.push_back(
                std::make_unique<Quadtree>(center, size / 2, level + 1));
        }
    }

    double Quadtree::get_variance() {
        std::vector<double> weights;
        for (auto &child : children) {
            recur_weights(*child, weights);
        }
        return variance(weights);
    }

    void Quadtree::recur_weights(Quadtree &root, std::vector<double> &weights) {
        if ((root.children).size()) {
            for (auto &child : root.children) {
                recur_weights(*child, weights);
            }
        } else {
            weights.push_back(root.weight);
        }
    }

    double variance(std::vector<double> &values) {
        int n = values.size();
        if (n == 0) {
            return 0;
        }

        double mean = 0;
        for (auto &v : values) {
            mean += v;
        }
        mean /= n;

        double var = 0;
        for (auto &v : values) {
            double d = v - mean;
            var += d * d;
        }
        return var / n;
    }
} // namespace HyperNEAT
