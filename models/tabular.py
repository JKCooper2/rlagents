import numpy as np
from collections import defaultdict


class TabularModel:
    def __init__(self, n_actions, observation_fa, mean=0.0, std=1.0):
        self.n_actions = n_actions
        self.n_observations = observation_fa.num_discrete
        self.mean = mean
        self.std = std

        self.weights = None
        self.keys = None

        self.reset()

    def state_value(self, observation):
        return max(self.weights[observation])

    def action_value(self, observation):
        return self.weights[observation]

    def action(self, observation):
        return np.argmax(self.weights[observation])

    def export_values(self):
        values = []

        for i in range(len(self.weights)):
            values.extend(self.weights[i])

        return np.array(values)

    def import_values(self, values):
        for i in range(len(values)/self.n_actions):
            self.weights[i] = np.array(values[self.n_actions * i: self.n_actions * i + self.n_actions])

    def reset(self):
        self.weights = np.random.normal(self.mean, scale=self.std, size=(self.n_observations, self.n_actions))
        self.keys = None
