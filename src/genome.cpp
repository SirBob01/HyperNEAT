#include "genome.h"

namespace HyperNEAT {
    bool Edge::operator==(const Edge &other) const {
        return from == other.from && to == other.to;
    }

    std::size_t EdgeHash::operator()(Edge const &s) const noexcept {
        std::size_t h1 = std::hash<int>{}(s.from);
        std::size_t h2 = std::hash<int>{}(s.to);
        return h1 ^ (h2 << 1);
    }

    Genome::Genome(GenomeParameters params) {
        _inputs = 4;
        _outputs = 1;

        _params = params;
        _fitness = 0.0;

        // Initialize the CPPN with minimal topology
        // Weights are random and biases are zero-initialized
        for (int i = 0; i < _inputs; i++) {
            auto random_it = std::next(std::begin(activations),
                                       randrange(0, activations.size()));
            _nodes.push_back({0, random_it->second});
        }
        for (int i = 0; i < _outputs; i++) {
            _nodes.push_back({0, tanh});
        }

        // Connect all inputs to all outputs
        for (int i = 0; i < _inputs; i++) {
            for (int j = _inputs; j < _inputs + _outputs; j++) {
                add_edge({i, j});
            }
        }

        // Generate the initial network structure
        update_structure();
    }

    Genome::Genome(std::vector<NodeGene> &nodes,
                   std::unordered_map<Edge, EdgeGene, EdgeHash> &edges,
                   GenomeParameters params) {
        _inputs = 4;
        _outputs = 1;

        _params = params;
        _fitness = 0.0;

        _nodes = nodes;
        _edges = edges;

        // Generate network structure from provided nodes and edges
        update_structure();
    }

    Genome::Genome(const Genome &genome) {
        _inputs = genome._inputs;
        _outputs = genome._outputs;

        _params = genome._params;
        _fitness = genome._fitness;

        _nodes = genome._nodes;
        _edges = genome._edges;

        _adjacency = genome._adjacency;
        _sorted = genome._sorted;
    }

    Genome &Genome::operator=(const Genome &genome) {
        _inputs = genome._inputs;
        _outputs = genome._outputs;

        _params = genome._params;
        _fitness = genome._fitness;

        _nodes = genome._nodes;
        _edges = genome._edges;

        _adjacency = genome._adjacency;
        _sorted = genome._sorted;
        return *this;
    }

    Genome::Genome(const Genome &a, const Genome &b) {
        _inputs = 4;
        _outputs = 1;

        _params = a._params;
        _fitness = 0.0;

        const auto &a_edges = a.get_edges();
        const auto &b_edges = b.get_edges();

        const auto &a_nodes = a.get_nodes();
        const auto &b_nodes = b.get_nodes();

        // Calculate disjoint and matching edges
        std::vector<Edge> matching;
        std::vector<Edge> a_disjoint;
        std::vector<Edge> b_disjoint;
        for (auto &e : b_edges) {
            if (a_edges.count(e.first) == 0) {
                b_disjoint.push_back(e.first);
            } else {
                matching.push_back(e.first);
            }
        }
        for (auto &e : a_edges) {
            if (b_edges.count(e.first) == 0) {
                a_disjoint.push_back(e.first);
            }
        }

        // Inherit matching genes from random parents
        for (auto &edge : matching) {
            if (random() < 0.5) {
                _edges[edge] = a_edges.at(edge);
            } else {
                _edges[edge] = b_edges.at(edge);
            }
            if (!a_edges.at(edge).enabled || !b_edges.at(edge).enabled) {
                if (random() < _params.crossover_gene_disable_rate) {
                    _edges[edge].enabled = false;
                } else {
                    _edges[edge].enabled = true;
                }
            }
        }

        // Inherit disjoint genes from fitter parent
        if (a.get_fitness() > b.get_fitness()) {
            for (auto &edge : a_disjoint) {
                _edges[edge] = a_edges.at(edge);
            }
        } else {
            for (auto &edge : b_disjoint) {
                _edges[edge] = b_edges.at(edge);
            }
        }

        // Inherit nodes from fitter parent
        int max_node = 0;
        for (auto &e : _edges) {
            max_node = std::max(max_node, e.first.from);
            max_node = std::max(max_node, e.first.to);
        }
        for (int n = 0; n <= max_node; n++) {
            if (a.get_fitness() > b.get_fitness()) {
                _nodes.push_back(a_nodes[n]);
            } else {
                _nodes.push_back(b_nodes[n]);
            }
        }

        // Generate network structure from inherited nodes and edges
        update_structure();
    }

    void Genome::topological_sort(int node, std::unordered_set<int> &visited) {
        if (visited.count(node)) {
            return;
        }

        for (int &adj : _adjacency[node]) {
            topological_sort(adj, visited);
        }

        visited.insert(node);
        _sorted.push_back(node);
    }

    void Genome::update_structure() {
        // Update the adjacency list
        _adjacency.clear();
        _adjacency.resize(_nodes.size());
        for (auto &p : _edges) {
            Edge e = p.first;
            _adjacency[e.to].push_back(e.from);
        }

        // Update the topological sort of the nodes
        int n = _nodes.size();
        _sorted.clear();
        std::unordered_set<int> visited;
        for (int i = 0; i < n; i++) {
            topological_sort(i, visited);
        }
    }

    void Genome::add_node(Edge edge) {
        int new_node = _nodes.size();
        _edges[edge].enabled = false;
        _edges[{edge.from, new_node}] = {1.0, true};
        _edges[{new_node, edge.to}] = {_edges[edge].weight, true};

        auto random_it = std::next(std::begin(activations),
                                   randrange(0, activations.size()));
        _nodes.push_back({0, random_it->second});
    }

    bool Genome::add_edge(Edge edge) {
        if (_edges.count(edge)) {
            return false;
        }
        _edges[edge] = {random() * 2.0 - 1.0, true};
        return true;
    }

    void Genome::toggle_enable(Edge edge) {
        _edges[edge].enabled = !_edges[edge].enabled;
    }

    bool Genome::shift_weight(Edge edge) {
        if (!_edges[edge].enabled) {
            return false;
        }
        _edges[edge].weight += ((1.0 - random()) * _params.mutation_power * 2) -
                               _params.mutation_power;
        return true;
    }

    bool Genome::reset_weight(Edge edge) {
        if (!_edges[edge].enabled) {
            return false;
        }

        _edges[edge].weight = random() * 2.0 - 1.0;
        return true;
    }

    void Genome::shift_bias(int node) {
        _nodes[node].bias += ((1.0 - random()) * _params.mutation_power * 2) -
                             _params.mutation_power;
    }

    void Genome::reset_bias(int node) {
        _nodes[node].bias = random() * 2.0 - 1.0;
    }

    void Genome::change_activation(int node) {
        auto random_it = std::next(std::begin(activations),
                                   randrange(0, activations.size()));
        _nodes[node].function = random_it->second;
    }

    double Genome::forward(Point p0, Point p1) {
        // Map the 2D points to the input vector
        _nodes[0].activation = p0.x;
        _nodes[1].activation = p0.y;
        _nodes[2].activation = p1.x;
        _nodes[3].activation = p1.y;

        // Forward propagation
        for (int i = _inputs; i < _sorted.size(); i++) {
            int to = _sorted[i];
            double sum = 0.0;
            for (int from : _adjacency[to]) {
                EdgeGene &gene = _edges[{from, to}];
                if (gene.enabled) {
                    sum += gene.weight * _nodes[from].activation;
                }
            }
            _nodes[to].activation = _nodes[to].function(sum + _nodes[to].bias);
        }
        return _nodes[_inputs].activation;
    }

    void Genome::mutate() {
        bool error;
        FunctionalDistribution dist = {
            {_params.m_weight_shift,
             [&]() {
                 auto random_edge =
                     std::next(std::begin(_edges), randrange(0, _edges.size()));
                 error = !shift_weight(random_edge->first);
             }},
            {_params.m_weight_change,
             [&]() {
                 auto random_edge =
                     std::next(std::begin(_edges), randrange(0, _edges.size()));
                 error = !reset_weight(random_edge->first);
             }},
            {_params.m_bias_shift,
             [&]() {
                 int random_node = randrange(_inputs, _nodes.size());
                 shift_bias(random_node);
             }},
            {_params.m_bias_change,
             [&]() {
                 int random_node = randrange(_inputs, _nodes.size());
                 reset_bias(random_node);
             }},
            {_params.m_node,
             [&]() {
                 auto random_edge =
                     std::next(std::begin(_edges), randrange(0, _edges.size()));
                 add_node(random_edge->first);
             }},
            {_params.m_edge,
             [&]() {
                 // Generate random nodes (i, j) such that:
                 // 1. i is not an output
                 // 2. j is not an input
                 // 3. i < j in the topological sort
                 int n = _nodes.size();
                 int i = randrange(0, n - _outputs);
                 int j = randrange(std::max(_inputs, i + 1), n);
                 error = !add_edge({_sorted[i], _sorted[j]});
             }},
            {_params.m_toggle_enable,
             [&]() {
                 auto random_edge =
                     std::next(std::begin(_edges), randrange(0, _edges.size()));
                 toggle_enable(random_edge->first);
             }},
            {_params.m_activation,
             [&]() {
                 int random_node = randrange(_inputs + _outputs, _nodes.size());
                 if (random_node >= _nodes.size()) {
                     error = true;
                 } else {
                     change_activation(random_node);
                 }
             }},
        };
        do {
            error = false;
            probability_function(dist);
        } while (error);

        update_structure();
    }

    const std::vector<NodeGene> &Genome::get_nodes() const { return _nodes; }

    const std::unordered_map<Edge, EdgeGene, EdgeHash> &
    Genome::get_edges() const {
        return _edges;
    }

    double Genome::distance(const Genome &other) const {
        const auto &a_edges = _edges;
        const auto &b_edges = other._edges;

        const auto &a_nodes = _nodes;
        const auto &b_nodes = other._nodes;

        int n_edges = std::max(a_edges.size(), b_edges.size());
        int disjoint = 0;
        for (auto &e : b_edges) {
            if (a_edges.count(e.first) == 0) {
                disjoint++;
            }
        }
        for (auto &e : a_edges) {
            if (b_edges.count(e.first) == 0) {
                disjoint++;
            }
        }

        double t1 = static_cast<double>(disjoint) / n_edges;
        double t2 = 0;
        double t3 = 0;

        // For matching edges, calculate the average weight difference
        int matching = 0;
        for (auto &e : a_edges) {
            if (b_edges.count(e.first)) {
                double w_diff = b_edges.at(e.first).weight - e.second.weight;
                t2 += std::fabs(w_diff);
                matching++;
            }
        }
        if (matching) {
            t2 /= matching;
        }

        // For matching nodes, find the total mismatched activation functions
        int n_nodes = std::min(a_nodes.size(), b_nodes.size());
        for (int i = 0; i < n_nodes; i++) {
            if (a_nodes[i].activation != b_nodes[i].activation) {
                t3 += 1;
            }
        }
        t3 /= n_nodes;

        return t1 * _params.c1 + t2 * _params.c2 + t3 * _params.c3;
    }

    double Genome::get_fitness() const { return _fitness; }

    void Genome::set_fitness(double fitness) { _fitness = fitness; }
} // namespace HyperNEAT