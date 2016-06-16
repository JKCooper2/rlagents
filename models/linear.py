import numpy as np


class LinearModel:
    def __init__(self, n_obs, bias=False):
        self.values = np.random.randn(n_obs)
        self.n_obs = n_obs + int(bias)
        self.bias = bias

    def sum_obs(self, observation):
        return sum(observation[i] * self.values[i] for i in range(len(observation))) + int(self.bias) * self.values[-1]

    def score(self, observation):
        return self.sum_obs(observation)

    def export_values(self):
        return self.values

    def import_values(self, values):
        if len(values) != self.n_obs:
            raise ValueError

        self.values = values

    def reset(self):
        self.values = np.random.randn(self.n_obs)


class BinaryLinearModel(LinearModel):
    def __init__(self, n_obs, bias=False):
        LinearModel.__init__(self, n_obs, bias)

    def score(self, observation):
        return self.sum_obs(observation) >= 0
