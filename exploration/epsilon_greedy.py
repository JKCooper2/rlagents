import numpy as np
from rlagents import validate, defaults


class EpsilonGreedy:
    def __init__(self, action_space, decay=None):
        self._action_space = action_space
        self._decay = decay if decay is not None else defaults.epsilongreedy_decay()

        validate.decay(self._decay)
        validate.action_space(self._action_space)

    @property
    def epsilon(self):
        return self._decay.value

    def choose_action(self, model, observation):
        validate.model(model)

        if np.random.uniform() < self._decay.value:
            return self._action_space.sample()

        action = model.action(observation)

        validate.action(self._action_space, action)

        return action

    def update(self):
        self._decay.update()
