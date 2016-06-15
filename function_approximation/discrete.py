import numpy as np


class Discrete:
    def __init__(self, values):
        self.values = values
        self.max = np.prod(self.values)

    def __validate(self, observation):
        for i in range(len(self.values)):
            assert observation[i] < self.values[i]

    def to_array(self, observation):
        if len(self.values) == 1:
            observation = [observation]

        self.__validate(observation)

        array_val = 0

        for i, obs in enumerate(observation):
            array_val += obs * max(np.prod(self.values[i+1:]), 1)

        return int(array_val)
