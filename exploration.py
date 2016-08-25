import warnings
import numpy as np

from rlagents.functions.decay import DecayBase, FixedDecay

"""
Exploration biases the action-values returned by a model (in the form of an array or in future a distribution)
"""


class ExplorationBase(object):
    model = None

    def update(self):
        raise NotImplementedError

    def bias_action_value(self, observation):
        raise NotImplementedError


class DefaultExploration(ExplorationBase):
    def update(self):
        pass

    def bias_action_value(self, observation):
        return self.model.action_value(observation)


class EpsilonGreedy(ExplorationBase):
    def __init__(self, action_space, decay=None):
        self.action_space = action_space
        self.decay = decay

    @property
    def value(self):
        return self.decay.value

    @property
    def decay(self):
        return self._decay

    @decay.setter
    def decay(self, d):
        if not isinstance(d, DecayBase):
            d = FixedDecay(0.1, 0, 0.1)
            warnings.warn("Decay type invalid, using default. {0}".format(d))

        self._decay = d

    def __str__(self):
        return "EpsilonGreedy decay: {0}".format(self.decay)

    def bias_action_value(self, observation):
        q_s = self.model.action_value(observation)

        if np.random.uniform() < self.value:
            # Select a random action and max it the best action
            q_s[np.random.randint(len(q_s))] = max(q_s) + 1

        return q_s

    def update(self):
        self.decay.update()


class Softmax(ExplorationBase):
    """
    Softmax selects the action to take based on the actions value relative to other actions
    Negative values are very unlikely to be chosen
    """
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, t):
        if t < 0 or t > 1:
            raise ValueError("Temperature must be between 0 and 1 inclusive")

        self._temperature = t

    def bias_action_value(self, observation):
        q_s = self.model.action_value(observation)

        probabilities = []

        for action in range(len(q_s)):
            numerator = np.e ** (q_s[action]/self.temperature)
            denominator = sum([np.e ** (q/self.temperature) for q in q_s])
            chance = numerator / denominator

            probabilities.append(chance)

        choice = np.random.uniform()
        cum_sum = 0

        action = None

        for act, value in enumerate(probabilities):
            cum_sum += value

            if cum_sum >= choice:
                q_s[act] = max(q_s) + 1
                break

        return action

    def update(self):
        pass
