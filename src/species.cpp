#include "species.h"

namespace HyperNEAT {
    Specie::Specie(std::unique_ptr<Genome> &&representative,
                   NEATParameters params) {
        _members.push_back(std::move(representative));
        _params = params;
        _stagnation_count = 0;
        _fitness_sum = 0;
    }

    Specie::Specie(const Specie &other) {
        for (auto &genome : other._members) {
            _members.push_back(std::make_unique<Genome>(*genome));
        }

        _params = other._params;
        _stagnation_count = other._stagnation_count;

        _fitness_sum = other._fitness_sum;
        _fitness_history = other._fitness_history;
    }

    void Specie::add(std::unique_ptr<Genome> &&genome) {
        _members.push_back(std::move(genome));
    }

    Genome &Specie::get_repr() { return *_members[0]; }

    std::vector<std::unique_ptr<Genome>> &Specie::get_members() {
        return _members;
    }

    void Specie::update_fitness() {
        int n = _members.size();
        _fitness_sum = 0;
        for (auto &genome : _members) {
            _fitness_sum += genome->get_fitness() / n;
        }
        _fitness_history.push(_fitness_sum);
        if (_fitness_history.size() > _params.max_stagnation) {
            _fitness_history.pop();
        }
    }

    double Specie::get_fitness_sum() { return _fitness_sum; }

    Genome &Specie::sample() {
        int index = randrange(0, _members.size());
        return *_members[index];
    }

    Genome &Specie::get_best() {
        Genome &best = *_members[0];
        for (auto &genome : _members) {
            if (genome->get_fitness() > best.get_fitness()) {
                best = *genome;
            }
        }
        return best;
    }

    void Specie::cull() {
        std::sort(_members.begin(),
                  _members.end(),
                  [](const auto &a, const auto &b) {
                      return a->get_fitness() > b->get_fitness();
                  });
        int new_size = 1 + (1 - _params.cull_percent) * _members.size();
        _members.resize(new_size);
    }

    bool Specie::can_progress() {
        int n = _fitness_history.size();
        if (n < _params.max_stagnation) {
            return true;
        }

        // Check if the overall average total fitness has improved over time
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