import numpy as np
import warnings

from rlagents.functions.decay import DecayBase, FixedDecay


class EvolutionaryBase(object):
    def next_generation(self, batch, results):
        raise NotImplementedError


class CrossEntropy(EvolutionaryBase):
    """
    Cross Entropy evolution selects the top x% of samples in a batch
    and generates a new batch from the mean/std of the those samples
    """

    def __init__(self, elite=0.2):
        """
        :param elite: float
            Percentage of samples selected to generate next batch.
            Must be between 0 and 1
        """
        self.elite = elite

    @property
    def elite(self):
        return self._elite

    @elite.setter
    def elite(self, e):
        if e < 0 or e > 1:
            raise ValueError("Elite must be between 0 and 1 inclusive")

        self._elite = e

    def next_generation(self, batch, results):
        """
        Selects the top x% samples from the batch and generates
        a new batch from the mean/std of the those samples
        :param batch: list of models
        :param results: list of rewards for each model
        :return: list of models
        """

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
    """
    Genetic Algorithm works by selecting two parent samples, with better performing samples
    having increased likelihood of being selected, and then swapping model values, along with a
    (normally) small chance of that value being mutated.
    """

    def __init__(self, crossover=0.3, mutation_rate=0.1, mutation_amount=0.02, scaling=1):
        """

        :param crossover: float
            Likelihood of switching parents to get the 'genes' from
            Range (0, 1)
        :param mutation_rate: float
            Likelihood of the current gene being mutated
            Range (0, 1)
        :param mutation_amount: float
            Percent to generate the standard deviation of the mutation from
            Range (0, 3)
        :param scaling: float
            How much results are scaled so high scores appear a lot better than low scores
            1 means no scaling applied
            Range (0, 5)
        """
        self.crossover = crossover
        self.mutation_rate = mutation_rate
        self.mutation_amount = mutation_amount
        self.scaling = scaling

    @property
    def crossover(self):
        return self._crossover

    @crossover.setter
    def crossover(self, c):
        if c < 0 or c > 1:
            raise ValueError("Crossover must be between 0 and 1 inclusive")

        self._crossover = c

    @property
    def mutation_rate(self):
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, mr):
        if mr <= 0 or mr >= 1:
            raise ValueError("Mutation Rate must be between 0 and 1")

        self._mutation_rate = mr

    @property
    def mutation_amount(self):
        return self._mutation_amount

    @mutation_amount.setter
    def mutation_amount(self, ma):
        if ma <= 0 or ma > 3:
            raise ValueError("Mutation Amount must be between 0 and 3")

        self._mutation_amount = ma

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, s):
        if s < 0 or s > 5:
            raise ValueError("Scaling must be between 0 and 5 inclusive")

        self._scaling = s

    def _scale_results(self, results):
        """
        Normalises results within range [0, 1] and applies scaling
        :param results: list of floats
        :return: list of floats
        """
        result_max = max(results)  # Stops numbers growing too large to handle
        result_min = min(results) - 0.001  # Stops result ranges that cover 0 causing issues with odd numbers scalings)
        return [(r - result_min) ** self.scaling / (result_max - result_min) ** self.scaling for r in results]

    def next_generation(self, batch, results):
        """
        :param batch: list of x models
        :param results: list of x floats
        :return: list of x models
        """
        results = self._scale_results(results)

        gen_size = len(batch)
        batch_vals = [b.export_values() for b in batch]
        next_gen = []

        for i in range(gen_size):
            father = self._roulette(results)
            mother = self._roulette(results)

            child = self._create_child(batch_vals[father], batch_vals[mother])

            next_gen.append(child)

        for i in range(len(batch)):
            batch[i].import_values(np.array(next_gen[i]))

        return batch

    def _create_child(self, father, mother):
        """
        Selected genes from both the father and mother with some mutation
        :param father: list of x floats
        :param mother: list of x floats
        :return: list of x floats
        """
        current_sel = 1

        child = []

        for j in range(len(father)):
            if np.random.uniform() <= self.crossover:
                current_sel *= -1

            if current_sel == 1:
                child.append(father[j])

            else:
                child.append(mother[j])

            if np.random.uniform() < self.mutation_rate:
                child[j] = np.random.normal(child[j], abs(child[j] * self.mutation_amount))

        return child

    @staticmethod
    def _roulette(results):
        """
        Select index based on score
        :param results: list of floats
        :return: int
            Index of parent to choose
        """
        selection = np.random.uniform() * sum(results)
        cum_sum = 0

        for i, score in enumerate(results):
            cum_sum += score
            if cum_sum >= selection:
                return i


class HillClimbing(EvolutionaryBase):
    """
    Hill Climbing works by selecting the best sample from a batch, then
    generating a new batch from that best sample
    """
    def __init__(self, spread=None, accept_equal=False):
        """
        :param spread: functions.DecayBase
            Used to determine standard deviation of new best sample
        :param accept_equal: boolean
            Whether equal best will replace current best
        """
        self.accept_equal = accept_equal
        self.spread = spread

        self.best_weights = None
        self.best_score = None

    @property
    def spread(self):
        return self._spread

    @spread.setter
    def spread(self, s):
        if s is None:
            s = FixedDecay(0.05, 0, 0)

        if not isinstance(s, DecayBase):
            raise TypeError("Spread not a valid DecayBase")

        self._spread = s

    def is_new_best(self, best):
        if self.best_weights is None:
            return True

        if best > self.best_score:
            return True

        if best == self.best_score and self.accept_equal:
            return True

        return False

    def next_generation(self, batch, results):
        """
        Finds the best sample
        Updates the current best if needed
        Creates new batch based on current best

        :param batch: list of Models
        :param results: list of floats
        :return: list of Models
        """
        best = np.argmax(results)

        if self.is_new_best(results[best]):
            self.best_weights = batch[best].export_values().copy()
            self.best_score = results[best]

        for i in range(len(batch)):
            child = []
            for j in range(len(self.best_weights)):
                std = max(abs(self.best_weights[j] * self.spread.value), 0.001)
                child.append(np.random.normal(self.best_weights[j], std))

            batch[i].import_values(np.array(child).copy())

        self.spread.update()

        return batch


class SimulatedAnnealing(EvolutionaryBase):
    """
    Simulated Annealing evolves a single sample by modifying one of it's values it and accepting the
    updated value with a reducing probability. When it starts (temperature is high)
    a lower value may be selected, but this reduces over time.

    This is based on the process of hardening metal.
    """
    def __init__(self, temperature=None, shift=None):
        """
        :param temperature: functions.Decay
            Influences likelihood of new sample being accepted
        :param shift: functions.Decay
            Amount to alter the randomly selected value by
        """
        self.temperature = temperature
        self.shift = shift

        self.testing_vals = None
        self.testing_result = 0

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, t):
        if t is None:
            t = FixedDecay(10, decay=0.997, minimum=0.1)

        if not isinstance(t, DecayBase):
            raise TypeError("Temperature must be of type DecayBase")

        self._temperature = t

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, s):
        if s is None:
            s = FixedDecay(1, decay=0.995, minimum=0.01)

        if not isinstance(s, DecayBase):
            raise TypeError("Shift must be of type DecayBase")

        self._shift = s

    @property
    def testing_vals(self):
        if self._testing_vals is None:
            return None

        return self._testing_vals.copy()

    @testing_vals.setter
    def testing_vals(self, t):
        if t is None:
            self._testing_vals = None

        else:
            self._testing_vals = t.copy()

    def next_generation(self, batch, results):
        if len(batch) > 1:
            warnings.warn("Simulated Annealing should use batch size of 1")

        batch_vals = batch[0].export_values()
        result = results[0]

        if self.testing_vals is None:
            self.testing_vals = batch_vals
            self.testing_result = result

        acceptance_probability = np.e ** ((result - self.testing_result) / self.temperature.value)
        # print self.testing_result, result, self.shift.value, self.temperature.value, acceptance_probability

        if np.random.uniform() < acceptance_probability:
            self.testing_vals = batch_vals
            self.testing_result = result

        batch[0].import_values(self.__create_next_test(self.testing_vals))

        self.temperature.update()
        self.shift.update()

        return batch

    def __create_next_test(self, batch_vals):
        choice = np.random.randint(len(batch_vals))

        batch_vals[choice] = np.random.normal(batch_vals[choice], abs(batch_vals[choice] * self.shift.value))

        return batch_vals
