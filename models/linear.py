import numpy as np


class LinearModel:
    def __init__(self, n):
        self.values = np.random.randn(n)

    def score(self, observation):
        return sum(observation[i] * self.values[i] for i in range(len(observation))) >= 0
