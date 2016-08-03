import numpy as np
from rlagents.functions.decay import DecayBase, FixedDecay


class EpsilonGreedy(object):
    def __init__(self, action_space, decay=None):
        self.action_space = action_space
        self.decay = decay if decay is not None else FixedDecay(0.1, 0, 0.1)

    @property
    def value(self):
        return self.decay.value

    @property
    def decay(self):
        return self._decay

    @decay.setter
    def decay(self, d):
        if not isinstance(d, DecayBase):
            raise TypeError("Decay must be a sub-class of DecayBase")

        self._decay = d

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, a):
        self._action_space = a

    def choose_action(self, model, observation):
        if np.random.uniform() < self.value:
            return self.action_space.sample()

        action = model.action(observation)

        return action

    def update(self):
        self.decay.update()
