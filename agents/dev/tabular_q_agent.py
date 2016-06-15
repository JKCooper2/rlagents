import numpy as np
from collections import defaultdict
from rlagents.function_approximation.dev.tiles import SingleTiling
from rlagents.function_approximation.dev.discrete import Discrete
from gym.spaces import discrete, tuple_space, box


class TabularQAgent(object):
    def __init__(self, action_space, observation_space, init_mean=0.0, init_std=0.2, alpha=0.5, epsilon=1, discount=0.95):
        self.name = "TabularQAgent"
        self.alg_id = "alg_OwSFZtRR2eZYkcxkG74Q"
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = self.action_space.n

        self.init_mean = init_mean
        self.init_std = init_std
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

        self.epsilon_decay = 0.997  # 0.997 = 5% per 1000 eps
        self.epsilon_min = 0.02

        self.alpha_decay = 0.996    # 0.996 = 2% per 1000 eps
        self.alpha_min = 0.02

        self.step_cost = -0.01  # So agent doesn't like states it's already been in that haven't lead to a reward

        self.prev_obs = None
        self.prev_action = None

        self.fa = self.__set_fa()

        self.ep_reward = 0
        self.ep_count = 0

        self.q = defaultdict(lambda: self.init_std * np.random.randn(self.action_n) + self.init_mean)

    def __set_fa(self):
        if isinstance(self.observation_space, tuple_space.Tuple):
            return Discrete([space.n for space in self.observation_space.spaces])

        elif isinstance(self.observation_space, box.Box):
            return SingleTiling(self.observation_space, 2)

        elif isinstance(self.observation_space, discrete.Discrete):
            return Discrete([self.observation_space.n])

    # Epsilon Greedy
    def __choose_action(self, observation):
        return np.argmax(self.q[observation]) if np.random.random() > self.epsilon else self.action_space.sample()

    def __learn(self, observation, reward, done):
        future = np.max(self.q[observation]) if not done else 0.0
        self.q[self.prev_obs][self.prev_action] += self.alpha * (reward + self.discount * future - self.q[self.prev_obs][self.prev_action])

    def act(self, observation, reward, done):
        observation = self.fa.to_array(observation)

        reward += self.step_cost

        self.__learn(observation, reward, done)

        action = self.__choose_action(observation)

        self.ep_reward += reward
        self.prev_obs = observation
        self.prev_action = action

        if done:
            self.ep_count += 1
            self.ep_reward = 0

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if self.alpha > self.alpha_min:
                self.alpha *= self.alpha_decay

        return action
