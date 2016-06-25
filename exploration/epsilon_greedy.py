import numpy as np
from rlagents import validate
import rlagents
from gym.spaces.discrete import Discrete

class EpsilonGreedy:
    def __init__(self, action_space, decay=None):
        self._action_space = action_space
        self._decay = decay if decay is not None else rlagents.functions.decay.FixedDecay(0.1, 1, 0.1)

        validate.decay(self._decay)
        validate.action_space(self._action_space)

    @property
    def value(self):
        return self._decay.value

    def discrete_action_to_try(self, action_values):
        if type(self._action_space) is Discrete:
            tried = [av[0] for av in action_values]
            not_tried = [action for action in range(self._action_space.n) if action not in tried]
            return not_tried

        return []


    def choose_action(self, model, observation):
        validate.model(model)

        if np.random.uniform() < self.value:
            return self._action_space.sample()

        action_values = model.action_value(observation)

        if action_values is None:
            return self._action_space.sample()

        datt = self.discrete_action_to_try(action_values)
        if len(datt) > 0:
            action = np.random.choice(datt)

        else:
            action = action_values[0][0]

        try:
            validate.action(self._action_space, action)
        except:
            action = self._action_space.sample()

        return action

    def update(self):
        self._decay.update()
