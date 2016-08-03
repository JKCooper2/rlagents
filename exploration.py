import warnings
import numpy as np

from rlagents.functions.decay import DecayBase, FixedDecay


class ExplorationBase(object):
    def update(self):
        raise NotImplementedError

    def choose_action(self, model, observation):
        raise NotImplementedError


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

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, a):
        self._action_space = a

    def __str__(self):
        return "EpsilonGreedy decay: {0}".format(self.decay)

    def choose_action(self, model, observation):
        if np.random.uniform() < self.value:
            return self.action_space.sample()

        action = model.action(observation)

        return action

    def update(self):
        self.decay.update()


# Softmax selects the action to take based on the actions value relative to other actions
class Softmax(ExplorationBase):
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

    def choose_action(self, model, observation):
        q_s = model.action_value(observation)

        probabilities = []

        for action in range(len(q_s)):
            numerator = np.e ** (q_s[action]/self.temperature)
            denominator = sum([np.e ** (q_s[other]/self.temperature) for other in range(len(q_s))])
            chance = numerator / denominator

            probabilities.append(chance)

        choice = np.random.uniform()
        cum_sum = 0

        action = None

        for act, value in enumerate(probabilities):
            cum_sum += value

            if cum_sum >= choice:
                action = act
                break

        return action

    def update(self):
        return
