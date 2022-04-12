#include "species.h"

namespace HyperNEAT {
    Specie::Specie(Genome *representative, NEATParameters params) {
        _members.push_back(representative);
        _params = params;
        _stagnation_count = 0;
    }

    Specie::~Specie() {
        for (auto &genome : _members) {
            delete genome;
        }
    }

    void Specie::add(Genome *genome) { _members.push_back(genome); }

    Genome &Specie::get_repr() { return *_members[0]; }

    std::vector<Genome *> &Specie::get_members() { return _members; }

    void Specie::adjust_fitness() {
        int n = _members.size();
        _fitness_sum = 0;
        for (auto &genome : _members) {
            double fitness = genome->get_fitness();
            genome->set_adjusted_fitness(fitness / _members.size());
            _fitness_sum += genome->get_adjusted_fitness();
        }
        _fitness_history.push(_fitness_sum);
        if (_fitness_history.size() > _params.max_stagnation) {
            _fitness_history.pop();
        }
    }

    double Specie::get_fitness_sum() { return _fitness_sum; }

    Genome *Specie::sample() {
        int index = randrange(0, _members.size());
        return _members[index];
    }

    Genome *Specie::get_best() {
        Genome *best = _members[0];
        for (auto &genome : _members) {
            if (genome->get_fitness() > best->get_fitness()) {
                best = genome;
            }
        }
        return best;
    }

    void Specie::cull() {
        std::sort(_members.begin(), _members.end(), [](Genome *a, Genome *b) {
            return a->get_fitness() > b->get_fitness();
        });
        int index = 1 + (1 - _params.cull_percent) * _members.size();
        for (int i = index; i < _members.size(); i++) {
            delete _members[i];
        }
        while (_members.size() > index) {
            _members.pop_back();
        }
    }

    bool Specie::can_progress() {
        int n = _fitness_history.size();
        if (n < _params.max_stagnation) {
            return true;
        }
        double first = _fitness_history.front();
        double avg = 0;
        while (_fitness_history.size()) {
            avg += _fitness_history.front();
            _fitness_history.pop();
        }
        avg /= n;
        return avg > first;
    }
} // namespace HyperNEAT