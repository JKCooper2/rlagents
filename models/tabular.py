import numpy as np
from collections import defaultdict


class TabularModel:
    def __init__(self, action_space, observation_space, mean=0.0, std=1.0):
        self.action_space = action_space
        self.n_actions = action_space.n
        self.observation_space = observation_space
        self.mean = mean
        self.std = std

        self.weights = np.random.normal(mean, scale=std, size=(observation_space.n, self.n_actions))  # defaultdict(lambda: std * np.random.randn(self.n_actions) + mean)
        self.keys = None

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
            self.weights[i] = np.array(values[self.n_actions * i : self.n_actions * i + self.n_actions])

    def reset(self):
        self.weights = np.random.normal(self.mean, scale=self.std, size=(self.observation_space.n, self.n_actions))
        self.keys = None
