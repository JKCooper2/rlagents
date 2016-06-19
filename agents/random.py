import numpy as np


class RandomAgent(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = 'random'
        self.alg_id = "alg_MhPaN5c4TJOFS4tVFh8x3A"

    def act(self, observation, reward, done):
        return self.__validate_action(self.action_space.sample())

    def __validate_action(self, action):
        if hasattr(action, '__iter__'):
            for i in range(len(action)):
                self.__validate_action(action[i])
        elif np.isnan(action):
            action = np.random.normal(0, 1.0)

        return action

