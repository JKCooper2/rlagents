import numpy as np
import warnings

from rlagents.functions.decay import DecayBase, FixedDecay


class EvolutionaryBase(object):
    def next_generation(self, batch, results):
        raise NotImplementedError


class CrossEntropy(EvolutionaryBase):
    def __init__(self, elite=0.2):
        self.elite = elite  # Percentage of samples selected to generate next batch

    @property
    def elite(self):
        return self._elite

    @elite.setter
    def elite(self, e):
        if e < 0 or e > 1:
            raise ValueError("Elite must be between 0 and 1 inclusive")

        self._elite = e

    def next_generation(self, batch, results):
        elite_n = int(len(batch) * self.elite)

        batch_vals = [b.export_values() for b in batch]

        # Select the x best scoring samples based on reverse score order
        best = np.array([batch_vals[b] for b in np.argsort(results)[-elite_n:]])

        # Update the mean/std based on new values
        mean = best.mean(axis=0)
        std = best.std(axis=0)

        for i in range(len(batch)):
            batch[i].import_values(np.random.normal(mean, std))

        return batch


class GeneticAlgorithm(EvolutionaryBase):
    def __init__(self, crossover=0.3, mutation_rate=0.1, mutation_amount=0.02, scaling=None):
        self.crossover = crossover
        self.mutation_rate = mutation_rate
        self.mutation_amount = mutation_amount
        self.scaling = scaling  # How much results are scaled so high scores appear a lot better than low scores

    def next_generation(self, batch, results):
        if self.scaling is not None:
            result_max = max(results)   # Stops numbers growing too large to handle
            result_min = min(results) - 0.001  # Stops result ranges that cover 0 causing issues with odd numbers scalings)
            results = [(r - result_min) ** self.scaling / (result_max - result_min) ** self.scaling for r in results]

        gen_size = len(batch)
        total_score = sum(results)
        batch_vals = [b.export_values() for b in batch]
        next_gen = []

        for i in range(gen_size):
            father = self.__roulette(total_score, results)
            mother = self.__roulette(total_score, results)
            current_sel = 1

            child = []

            for j in range(len(batch_vals[0])):
                if np.random.uniform() <= self.crossover:
                    current_sel *= -1

                # Use father if current_sel is 1
                if current_sel == 1:
                    child.append(batch_vals[father][j])

                else:
                    child.append(batch_vals[mother][j])

                if np.random.uniform() < self.mutation_rate:
                    child[j] = np.random.normal(child[j], abs(child[j] * self.mutation_amount))

            next_gen.append(child)

        for i in range(len(batch)):
            batch[i].import_values(np.array(next_gen[i]))

        return batch

    @staticmethod
    def __roulette(total_score, results):
        selection = np.random.uniform() * total_score
        cum_sum = 0

        for i, score in enumerate(results):
            cum_sum += score
            if cum_sum >= selection:
                return i

        print selection, cum_sum
        raise ValueError("Couldn't select value using roulette")


class HillClimbing(EvolutionaryBase):
    def __init__(self, spread=None, accept_equal=False):
        self.accept_equal = accept_equal
        self.spread = spread

        self.best_weights = None
        self.best_score = None

    @property
    def spread(self):
        return self._spread

    @spread.setter
    def spread(self, s):
        if not isinstance(s, DecayBase):
            s = FixedDecay(0.05, 0, 0)
            warnings.warn("Decay type invalid, using default. {0}".format(s))

        self._spread = s

    def next_generation(self, batch, results):
        best = np.argmax(results)

        if self.best_weights is None or results[best] + int(self.accept_equal) > self.best_score:
            self.best_weights = batch[best].export_values().copy()
            self.best_score = results[best]

        for i in range(len(batch)):
            child = []
            for j in range(len(self.best_weights)):
                std = abs(self.best_weights[j] * self.spread.value) if abs(self.best_weights[j] * self.spread.value) > 0.001 else 0.001
                child.append(np.random.normal(self.best_weights[j], std))

            batch[i].import_values(np.array(child).copy())

        self.spread.update()

        return batch


class SimulatedAnnealing(EvolutionaryBase):
    def __init__(self, temperature=None, shift=None, bias=None):
        self.temperature = temperature
        self.shift = shift
        self.bias = bias

        self.testing_vals = None
        self.testing_result = 0

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, t):
        if not isinstance(t, DecayBase):
            t = FixedDecay(10, decay=0.997, minimum=0.1)
            warnings.warn("Decay type invalid, using default. {0}".format(t))

        self._temperature = t

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, b):
        if not isinstance(b, DecayBase):
            b = FixedDecay(1, decay=0.995, minimum=0.01)
            warnings.warn("Decay type invalid, using default. {0}".format(b))

        self._bias = b

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, s):
        if not isinstance(s, DecayBase):
            s = FixedDecay(1, decay=0.995, minimum=0.01)
            warnings.warn("Decay type invalid, using default. {0}".format(s))

        self._shift = s

    @property
    def testing_vals(self):
        return self._testing_vals

    @testing_vals.setter
    def testing_vals(self, t):
        if t is None:
            self._testing_vals = None

        else:
            self._testing_vals = t.copy()

    def next_generation(self, batch, results):
        if len(batch) > 1:
            print "Only the first sample is used in simulated annealing"

        batch_vals = batch[0].export_values()
        result = results[0]

        # No saved value to compare against
        if self.testing_vals is None:
            self.testing_vals = batch_vals
            self.testing_result = result

        acceptance_probability = np.e ** ((result - self.testing_result) / self.temperature.value)
        # print self.testing_result, result, self.shift.value, self.temperature.value, acceptance_probability

        if np.random.uniform() < acceptance_probability:
            self.testing_vals = batch_vals
            self.testing_result = result
            batch[0].import_values(self.__create_next_test(batch_vals, result))

        else:
            batch[0].import_values(self.__create_next_test(self.testing_vals.copy(), self.testing_result))

        self.temperature.update()
        self.bias.update()
        self.shift.update()

        return batch

    def __create_next_test(self, batch_vals, result):
        choice = np.random.randint(len(batch_vals))

        batch_vals[choice] = np.random.normal(batch_vals[choice], self.shift.value)

        if self.bias.value != 0:
            batch_vals[choice] += np.random.normal(0, self.bias.value)

        return batch_vals
