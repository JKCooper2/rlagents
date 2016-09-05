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

    def configure(self, model):
        self.model = model

    def export(self):
        raise NotImplementedError


class DefaultExploration(ExplorationBase):
    def update(self):
        pass

    def bias_action_value(self, observation):
        return self.model.action_value(observation)

    def export(self):
        return {"Type": "Default"}


class RandomExploration(ExplorationBase):
    def update(self):
        pass

    def bias_action_value(self, observation):
        if self.model.action_fa.space_type == 'D':
            q_s = self.model.action_value(observation)
            q_s[np.random.randint(len(q_s))] = max(q_s) + 1

            return q_s

        return self.model.action_fa.space.sample()

    def export(self):
        return {"Type": "Random"}


class EpsilonGreedy(ExplorationBase):
    def __init__(self, decay=None):
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
            d = FixedDecay(0.1, 1, 0.1)
            warnings.warn("Decay type invalid, using default. {0}".format(d))

        self._decay = d

    def __str__(self):
        return "EpsilonGreedy decay: {0}".format(self.decay)

    def bias_action_value(self, observation):
        q_s = self.model.action_value(observation).copy()   # Copied so doesn't alter weights in model

        if np.random.uniform() < self.value:
            # Select a random action and max it the best action
            q_s[np.random.randint(len(q_s))] = max(q_s) + 1

        return q_s

    def update(self):
        self.decay.update()

    def export(self):
        return {"Type": "Epsilon Greedy",
                "Decay": self.decay.export()}


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
        if not isinstance(t, DecayBase):
            raise ValueError("Temperature must be a valid DecayBase")

        if t.minimum < 0 or t.value < 0:
            raise ValueError("Temperature minimum and value must be greater than 0")

        self._temperature = t

    def bias_action_value(self, observation):
        q_s = self.model.action_value(observation).copy()

        probabilities = []

        for action in range(len(q_s)):
            numerator = np.e ** (q_s[action]/self.temperature.value)
            denominator = sum([np.e ** (q/self.temperature.value) for q in q_s])
            chance = numerator / denominator

            probabilities.append(chance)

        choice = np.random.uniform()
        cum_sum = 0

        for act, value in enumerate(probabilities):
            cum_sum += value

            if cum_sum >= choice:
                q_s[act] = max(q_s) + 1
                break

        return q_s

    def update(self):
        pass

    def export(self):
        return {"Type": "Softmax",
                "Temperature": self.temperature.export()}
