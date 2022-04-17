#include "pool.h"

namespace HyperNEAT {
    Pool::Pool(std::vector<Point> inputs,
               std::vector<Point> outputs,
               NEATParameters params) {
        _inputs = inputs;
        _outputs = outputs;
        _params = params;

        _generations = 0;

        // Register a population of random genomes
        for (int i = 0; i < params.population; i++) {
            add_genome(std::make_unique<Genome>(_params.genome_params));
        }

        // Pick a random genome as the initial global best
        _global_best = std::make_unique<Genome>(_species[0]->sample());
    }

    Pool::Pool(std::string filename, NEATParameters params) {
        _params = params;
        std::ifstream infile;
        infile.open(filename, std::ios::binary | std::ios::in);

        // Read input/output coordinates
        int input_count;
        infile.read(reinterpret_cast<char *>(&input_count), sizeof(int));
        for (int i = 0; i < input_count; i++) {
            Point input;
            infile.read(reinterpret_cast<char *>(&input), sizeof(Point));
            _inputs.push_back(input);
        }
        int output_count;
        infile.read(reinterpret_cast<char *>(&output_count), sizeof(int));
        for (int i = 0; i < output_count; i++) {
            Point output;
            infile.read(reinterpret_cast<char *>(&output), sizeof(Point));
            _outputs.push_back(output);
        }

        // Read generations
        infile.read(reinterpret_cast<char *>(&_generations), sizeof(int));

        // Read each genome
        int specie_count;
        infile.read(reinterpret_cast<char *>(&specie_count), sizeof(int));
        for (int i = 0; i < specie_count; i++) {
            int genome_count;
            infile.read(reinterpret_cast<char *>(&genome_count), sizeof(int));
            for (int j = 0; j < genome_count; j++) {
                add_genome(read_genome(infile));
            }
        }

        // Read the elites, hall of fame, and global best genomes
        int elite_count;
        infile.read(reinterpret_cast<char *>(&elite_count), sizeof(int));
        for (int i = 0; i < elite_count; i++) {
            _elites.push_back(read_genome(infile));
        }
        _global_best = read_genome(infile);
        infile.close();
    }

    void Pool::add_genome(std::unique_ptr<Genome> &&genome) {
        // Test if the new genome fits in an existing specie
        for (auto &specie : _species) {
            Genome &repr = specie->get_repr();
            if (repr.distance(*genome) < _params.distance_threshold) {
                specie->add(std::move(genome));
                return;
            }
        }

        // Create a new species
        _species.push_back(
            std::make_unique<Specie>(std::move(genome), _params));
    }

    void Pool::cull() {
        // Treat the elites as its own species
        for (auto &specie : _species) {
            for (auto &genome : specie->get_members()) {
                if (genome->get_fitness() > _global_best->get_fitness()) {
                    _global_best = std::make_unique<Genome>(*genome);
                }
            }
            _elites.push_back(std::make_unique<Genome>(specie->get_best()));
        }
        std::sort(_elites.begin(),
                  _elites.end(),
                  [](const auto &a, const auto &b) {
                      return a->get_fitness() > b->get_fitness();
                  });
        _elites.resize(
            std::min(static_cast<int>(_elites.size()), _params.target_species));

        // Cull stagnant species
        std::vector<std::unique_ptr<Specie>> survivors;
        for (auto &specie : _species) {
            if (specie->can_progress()) {
                specie->update_fitness();
                specie->cull();
                survivors.push_back(std::make_unique<Specie>(*specie));
            }
        }
        _species = std::move(survivors);
    }

    void Pool::repopulate() {
        int size = 0;
        for (auto &specie : _species) {
            size += (specie->get_members()).size();
        }
        if (!_species.size()) {
            // No specie survived; reset the population
            for (int i = 0; i < _params.population; i++) {
                double r = random();
                std::unique_ptr<Genome> genome;
                if (r < 0.5) {
                    genome = std::make_unique<Genome>(*_global_best);
                    genome->mutate();
                } else if (r < 0.75 && _elites.size() >= 2) {
                    Genome &mom = *_elites[randrange(0, _elites.size())];
                    Genome &dad = *_elites[randrange(0, _elites.size())];
                    if (&mom == &dad) {
                        genome = std::make_unique<Genome>(mom);
                        genome->mutate();
                    } else {
                        genome = std::make_unique<Genome>(mom, dad);
                    }
                } else {
                    genome = std::make_unique<Genome>(_params.genome_params);
                }
                add_genome(std::move(genome));
            }
            return;
        }
        while (size < _params.population) {
            double r = random();
            double cum_prob = 0;
            Specie &specie = sample_specie();
            std::unique_ptr<Genome> child;

            if (r < _params.crossover_probability) {
                Genome &mom = specie.sample();
                Genome &dad = specie.sample();
                if (&mom == &dad) {
                    child = std::make_unique<Genome>(mom);
                    child->mutate();
                } else {
                    child = std::make_unique<Genome>(mom, dad);
                }
            }
            cum_prob += _params.crossover_probability;

            if (r >= cum_prob && r < cum_prob + _params.mutation_probability) {
                Genome &parent = specie.sample();
                child = std::make_unique<Genome>(parent);
                child->mutate();
            }
            cum_prob += _params.mutation_probability;

            if (r >= cum_prob && r < cum_prob + _params.clone_probability) {
                Genome &parent = specie.sample();
                child = std::make_unique<Genome>(parent);
            }
            cum_prob += _params.clone_probability;
            assert(std::fabs(1.0 - cum_prob) < 0.001);
            add_genome(std::move(child));
            size++;
        }
    }

    Specie &Pool::sample_specie() {
        double r = random();
        double total = 0;
        for (auto &specie : _species) {
            total += specie->get_fitness_sum();
        }
        if (total == 0) {
            return *_species[randrange(0, _species.size())];
        }

        // Species with a higher adjusted fitness total are more likely to be
        // picked
        double cum_prob = 0;
        int i = 0;
        for (auto &specie : _species) {
            double prob = specie->get_fitness_sum() / total;
            i++;
            if (r >= cum_prob && r < cum_prob + prob) {
                return *specie;
            }
            cum_prob += prob;
        }
        return *_species[0];
    }

    std::unique_ptr<Genome> Pool::read_genome(std::ifstream &infile) {
        std::unordered_map<Edge, EdgeGene, EdgeHash> edges;
        std::vector<NodeGene> nodes;

        // Read the fitness
        double fitness;
        infile.read(reinterpret_cast<char *>(&fitness), sizeof(double));

        // Read the edges
        int edge_count;
        infile.read(reinterpret_cast<char *>(&edge_count), sizeof(int));
        for (int k = 0; k < edge_count; k++) {
            Edge edge;
            EdgeGene gene;
            infile.read(reinterpret_cast<char *>(&edge), sizeof(Edge));
            infile.read(reinterpret_cast<char *>(&gene), sizeof(EdgeGene));
            edges[edge] = gene;
        }

        // Read the nodes
        int node_count;
        infile.read(reinterpret_cast<char *>(&node_count), sizeof(int));
        for (int k = 0; k < node_count; k++) {
            NodeGene n;
            infile.read(reinterpret_cast<char *>(&n.bias), sizeof(double));

            // Read the function string
            int len;
            infile.read(reinterpret_cast<char *>(&len), sizeof(int));

            std::vector<char> tmp(len + 1);
            tmp[len] = 0;
            infile.read(&tmp[0], sizeof(char) * len);
            std::string function(&tmp[0]);

            // Set the remaining values
            n.function = activations.at(function);
            nodes.push_back(n);
        }
        std::unique_ptr<Genome> genome =
            std::make_unique<Genome>(nodes, edges, _params.genome_params);
        genome->set_fitness(fitness);
        return genome;
    }

    void Pool::write_genome(std::ofstream &outfile, Genome &genome) {
        auto edges = genome.get_edges();
        auto nodes = genome.get_nodes();

        // Save fitness
        double fitness = genome.get_fitness();
        outfile.write(reinterpret_cast<char *>(&fitness), sizeof(double));

        // Save the edges
        int edge_count = edges.size();
        outfile.write(reinterpret_cast<char *>(&edge_count), sizeof(int));
        for (auto &e : edges) {
            Edge edge = e.first;
            EdgeGene gene = e.second;
            outfile.write(reinterpret_cast<char *>(&edge), sizeof(Edge));
            outfile.write(reinterpret_cast<char *>(&gene), sizeof(EdgeGene));
        }

        // Save the nodes
        int node_count = nodes.size();
        outfile.write(reinterpret_cast<char *>(&node_count), sizeof(int));
        for (auto &n : nodes) {
            std::string function;
            for (auto &a : activations) {
                if (a.second == n.function) {
                    function = a.first;
                    break;
                }
            }
            outfile.write(reinterpret_cast<char *>(&n.bias), sizeof(double));

            // Save function string
            int len = function.length();
            outfile.write(reinterpret_cast<char *>(&len), sizeof(int));
            outfile.write(function.c_str(), sizeof(char) * len);
        }
    }

    void Pool::evolve(std::function<void(Phenome &phenome)> evaluator) {
        // Evaulate, cull, and repopulate
        for (auto &specie : _species) {
            for (auto &genome : specie->get_members()) {
                Phenome phenome(*genome,
                                _inputs,
                                _outputs,
                                _params.phenome_params);
                evaluator(phenome);
            }
        }
        cull();
        repopulate();

        // Dynamically update the distance threshold
        if (_species.size() < _params.target_species) {
            _params.distance_threshold -= _params.distance_threshold_delta;
        } else if (_species.size() > _params.target_species) {
            _params.distance_threshold += _params.distance_threshold_delta;
        }
        if (_params.distance_threshold < _params.distance_threshold_delta) {
            _params.distance_threshold = _params.distance_threshold_delta;
        }
        _generations++;
    }

    Phenome Pool::get_global_fittest() {
        return Phenome(*_global_best,
                       _inputs,
                       _outputs,
                       _params.phenome_params);
    }

    int Pool::get_generations() { return _generations; }

    void Pool::save(std::string filename) {
        std::ofstream outfile;
        outfile.open(filename, std::ios::binary | std::ios::out);

        // Save input/output coordinates
        int input_count = _inputs.size();
        outfile.write(reinterpret_cast<char *>(&input_count), sizeof(int));
        for (auto &input : _inputs) {
            outfile.write(reinterpret_cast<char *>(&input), sizeof(Point));
        }
        int output_count = _outputs.size();
        outfile.write(reinterpret_cast<char *>(&output_count), sizeof(int));
        for (auto &output : _outputs) {
            outfile.write(reinterpret_cast<char *>(&output), sizeof(Point));
        }

        // Save generations
        outfile.write(reinterpret_cast<char *>(&_generations), sizeof(int));

        // Save each genome
        int specie_count = _species.size();
        outfile.write(reinterpret_cast<char *>(&specie_count), sizeof(int));
        for (auto &specie : _species) {
            int genome_count = specie->get_members().size();
            outfile.write(reinterpret_cast<char *>(&genome_count), sizeof(int));
            for (auto &genome : specie->get_members()) {
                write_genome(outfile, *genome);
            }
        }

        // Save the elite genomes, hall of fame, and the global best
        int elite_count = _elites.size();
        outfile.write(reinterpret_cast<char *>(&elite_count), sizeof(int));
        for (auto &genome : _elites) {
            write_genome(outfile, *genome);
        }
        write_genome(outfile, *_global_best);
        outfile.close();
    }
} // namespace HyperNEAT