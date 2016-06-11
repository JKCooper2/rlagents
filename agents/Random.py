# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = 'random'

    def act(self, observation, reward, done):
        return self.action_space.sample()