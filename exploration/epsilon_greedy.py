import numpy as np
from rlagents.functions.decay import FixedDecay


class EpsilonGreedy:
    def __init__(self, action_space, epsilon=0.1, decay=1, minimum=0.01):
        self.action_space = action_space
        self.epsilon = FixedDecay(epsilon, decay, minimum)

    def choose_action(self, q_s):
        if np.random.uniform() < self.epsilon.value:
            return self.action_space.sample()

        return np.argmax(q_s)

    def update(self):
        self.epsilon.update()
