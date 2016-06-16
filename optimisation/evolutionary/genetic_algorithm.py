import numpy as np


class GeneticAlgorithm:
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
            batch[i].import_values(next_gen[i])

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