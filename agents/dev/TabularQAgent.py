import numpy as np
from collections import defaultdict
from rlagents.function_approximation.dev.tiles import SingleTiling


class TabularQAgent(object):
    def __init__(self, action_space, observation_space, init_mean=0.0, init_std=0.5, alpha=0.05, epsilon=0.1, discount=0.95):
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = self.action_space.n

        self.init_mean = init_mean
        self.init_std = init_std
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

        self.prev_obs = None
        self.prev_action = None

        self.fa = SingleTiling(self.observation_space, 8)

        self.ep_reward = 0
        self.ep_count = 0

        self.q = defaultdict(lambda: self.init_std * np.random.randn(self.action_n) + self.init_mean)

    # Epsilon Greedy
    def __choose_action(self, observation):
        return np.argmax(self.q[observation]) if np.random.random() > self.epsilon else self.action_space.sample()

    def __learn(self, observation, reward, done):
        future = np.max(self.q[observation]) if not done else 0.0
        self.q[self.prev_obs][self.prev_action] -= self.alpha * (self.q[self.prev_obs][self.prev_action] - reward - self.discount * future)

    def act(self, observation, reward, done):
        observation = self.fa.convert_base10(self.fa.get_tile(observation))

        self.__learn(observation, reward, done)

        action = self.__choose_action(observation)

        self.ep_reward += reward
        self.prev_obs = observation
        self.prev_action = action

        if done:
            self.ep_count += 1
            self.ep_reward = 0

        return action
