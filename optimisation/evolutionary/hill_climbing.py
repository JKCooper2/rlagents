import numpy as np
from rlagents.functions.decay import FixedDecay


class HillClimbing:
    def __init__(self, spread=None, accept_equal=False):
        self.accept_equal = accept_equal
        self.spread = spread if spread is not None else FixedDecay(0.05, 0, 0)

        self.best_weights = None
        self.best_score = None

    def next_generation(self, batch, results):
        best = np.argmax(results)

        if self.best_weights is None or results[best] + int(self.accept_equal) > self.best_score:
            self.best_weights = batch[best].export_values().copy()
            self.best_score = results[best]

        for i in range(len(batch)):
            child = []
            for j in range(len(self.best_weights)):
                child.append(np.random.normal(self.best_weights[j], abs(self.best_weights[j] * self.spread.value)))

            batch[i].import_values(np.array(child).copy())

        self.spread.update()

        return batch
