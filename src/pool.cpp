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
            Genome *g = new Genome(_params.genome_params);
            add_genome(g);
            if (_hall_of_fame.size() < _params.max_hall_of_fame) {
                _hall_of_fame.push_back(new Genome(*g));
            }
        }
        generate_phenomes();
        _global_best = new Genome(_species[0]->sample());
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
        int fame_count;
        infile.read(reinterpret_cast<char *>(&fame_count), sizeof(int));
        for (int i = 0; i < fame_count; i++) {
            _hall_of_fame.push_back(read_genome(infile));
        }
        _global_best = read_genome(infile);
        infile.close();
        generate_phenomes();
    }

    Pool::~Pool() {
        for (auto &specie : _species) {
            delete specie;
        }
        for (auto &elite : _elites) {
            delete elite;
        }
        for (auto &avatar : _hall_of_fame) {
            delete avatar;
        }
        for (auto &phenome : _phenomes) {
            delete phenome;
        }
        delete _global_best;
    }

    void Pool::add_genome(Genome *genome) {
        // Test if the new genome fits in an existing specie
        for (auto &specie : _species) {
            Genome &repr = specie->get_repr();
            if (repr.distance(*genome) < _params.distance_threshold) {
                specie->add(genome);
                return;
            }
        }

        // Create a new species
        _species.push_back(new Specie(genome, _params));
    }

    void Pool::generate_phenomes() {
        for (auto &phenome : _phenomes) {
            delete phenome;
        }
        _phenomes.clear();
        for (auto &s : _species) {
            for (auto &genome : s->get_members()) {
                _phenomes.push_back(new Phenome(*genome,
                                                _inputs,
                                                _outputs,
                                                _params.phenome_params));
            }
        }
    }

    void Pool::cull() {
        std::vector<Specie *> survivors;
        // Treat the elites as its own species
        for (auto &specie : _species) {
            for (auto &genome : specie->get_members()) {
                if (genome->get_fitness() > _global_best->get_fitness()) {
                    delete _global_best;
                    _global_best = new Genome(*genome);
                }
            }
            _elites.push_back(new Genome(specie->get_best()));
        }
        _hall_of_fame.push_back(new Genome(*_global_best));
        if (_hall_of_fame.size() > _params.max_hall_of_fame) {
            delete _hall_of_fame.front();
            _hall_of_fame.pop_front();
        }
        std::sort(_elites.begin(), _elites.end(), [](Genome *a, Genome *b) {
            return a->get_fitness() > b->get_fitness();
        });
        int index =
            std::min(static_cast<int>(_elites.size()), _params.target_species);
        for (int i = index; i < _elites.size(); i++) {
            delete _elites[i];
        }
        while (_elites.size() > index) {
            _elites.pop_back();
        }

        for (auto &specie : _species) {
            // Delete stagnant species
            if (specie->can_progress()) {
                specie->update_fitness();
                specie->cull();
                survivors.push_back(specie);
            } else {
                delete specie;
            }
        }
        _species = survivors;
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
                Genome *genome;
                if (r < 0.5) {
                    genome = new Genome(*_global_best);
                    genome->mutate();
                } else if (r < 0.75 && _elites.size() >= 2) {
                    Genome *mom = _elites[randrange(0, _elites.size())];
                    Genome *dad = _elites[randrange(0, _elites.size())];
                    if (mom == dad) {
                        genome = new Genome(*mom);
                        genome->mutate();
                    } else {
                        genome = new Genome(*mom, *dad);
                    }
                } else {
                    genome = new Genome(_params.genome_params);
                }
                add_genome(genome);
            }
            return;
        }
        while (size < _params.population) {
            double r = random();
            double cum_prob = 0;
            Genome *child = nullptr;
            Specie *specie = sample_specie();

            if (r < _params.crossover_probability) {
                Genome &mom = specie->sample();
                Genome &dad = specie->sample();
                if (&mom == &dad) {
                    child = new Genome(mom);
                    child->mutate();
                } else {
                    child = new Genome(mom, dad);
                }
            }
            cum_prob += _params.crossover_probability;

            if (r >= cum_prob && r < cum_prob + _params.mutation_probability) {
                Genome &parent = specie->sample();
                child = new Genome(parent);
                child->mutate();
            }
            cum_prob += _params.mutation_probability;

            if (r >= cum_prob && r < cum_prob + _params.clone_probability) {
                Genome &parent = specie->sample();
                child = new Genome(parent);
            }
            cum_prob += _params.clone_probability;
            assert(std::fabs(1.0 - cum_prob) < 0.001);
            add_genome(child);
            size++;
        }
    }

    Specie *Pool::sample_specie() {
        double r = random();
        double total = 0;
        for (auto &specie : _species) {
            total += specie->get_fitness_sum();
        }
        if (total == 0) {
            return _species[randrange(0, _species.size())];
        }

        // Species with a higher adjusted fitness sum are more likely to be
        // picked
        double cum_prob = 0;
        int i = 0;
        for (auto &specie : _species) {
            double prob = specie->get_fitness_sum() / total;
            i++;
            if (r >= cum_prob && r < cum_prob + prob) {
                return specie;
                break;
            }
            cum_prob += prob;
        }
        return nullptr;
    }

    Genome *Pool::read_genome(std::ifstream &infile) {
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

            char *tmp = new char[len + 1];
            tmp[len] = 0;
            infile.read(tmp, sizeof(char) * len);
            std::string function(tmp);
            delete[] tmp;

            // Set the remaining values
            n.function = activations.at(function);
            nodes.push_back(n);
        }
        Genome *genome = new Genome(nodes, edges, _params.genome_params);
        genome->set_fitness(fitness);
        return genome;
    }

    void Pool::write_genome(std::ofstream &outfile, Genome *genome) {
        auto edges = genome->get_edges();
        auto nodes = genome->get_nodes();

        // Save fitness
        double fitness = genome->get_fitness();
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

    std::vector<Phenome *> &Pool::get_phenomes() { return _phenomes; }

    void Pool::evolve() {
        cull();
        repopulate();
        generate_phenomes();

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

    int Pool::get_generations() { return _generations; }

    Phenome Pool::get_global_fittest() {
        return Phenome(*_global_best,
                       _inputs,
                       _outputs,
                       _params.phenome_params);
    }

    Phenome Pool::get_current_fittest() {
        Genome &genome = _species[0]->get_best();
        for (auto &specie : _species) {
            Genome &candidate = specie->get_best();
            if (candidate.get_fitness() > genome.get_fitness()) {
                genome = candidate;
            }
        }
        return Phenome(genome, _inputs, _outputs, _params.phenome_params);
    }

    std::vector<Phenome> Pool::get_hall_of_fame() {
        std::vector<Phenome> phenomes;
        for (auto &genome : _hall_of_fame) {
            phenomes.push_back(
                Phenome(*genome, _inputs, _outputs, _params.phenome_params));
        }
        return phenomes;
    }

    std::vector<Phenome> Pool::get_elites() {
        std::vector<Phenome> phenomes;
        for (auto &genome : _elites) {
            phenomes.push_back(
                Phenome(*genome, _inputs, _outputs, _params.phenome_params));
        }
        return phenomes;
    }

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
                write_genome(outfile, genome.get());
            }
        }

        // Save the elite genomes, hall of fame, and the global best
        int elite_count = _elites.size();
        outfile.write(reinterpret_cast<char *>(&elite_count), sizeof(int));
        for (auto &genome : _elites) {
            write_genome(outfile, genome);
        }
        int fame_count = _hall_of_fame.size();
        outfile.write(reinterpret_cast<char *>(&fame_count), sizeof(int));
        for (auto &genome : _hall_of_fame) {
            write_genome(outfile, genome);
        }
        write_genome(outfile, _global_best);
        outfile.close();
    }
} // namespace HyperNEAT