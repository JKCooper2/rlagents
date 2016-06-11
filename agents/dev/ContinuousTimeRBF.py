import itertools
import numpy as np

# From: https://gym.openai.com/algorithms/alg_jcLnjDqgSb2pPhtRFxMOIw


class RBFAgent:
    def __init__(self, action_space, observation_space, uniform=True, bins=2, discount=0.9, decay=0.9):
        self.name = "RBF Agent"
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_observations = self.observation_space.shape[0]

        self.uniform = uniform  # Distribution of RBF centroids, if false then random
        self.bins = bins  # Bins per dimension
        self.num_rbf = pow(self.bins, self.observation_space.shape[0])  # Number of RBFs
        self.discount = discount
        self.decay = decay  # Trace Decay, 0 = no traces
        self.var = pow(0.8 / (self.bins - 1), 2)

        self.learning_rate = 1
        self.epsilon = 1
        self.episode_count = 0
        self.ep_reward = 0
        self.prev_action = None
        self.prev_observation = None

        self.upper = np.array([0.5 if x == float("Inf") else x for x in self.observation_space.high])
        self.lower = np.array([-0.5 if x == -float("Inf") else x for x in self.observation_space.low])
        self.amp = self.upper - self.lower

        self.centroids = self.__initialise_centroids()
        self.weights = np.zeros((self.action_space.n, self.num_rbf + 1))
        self.trace = self.__add_bias(np.zeros(self.num_rbf))

    def __initialise_centroids(self):
        if self.uniform:
            return np.array(list(itertools.product(np.linspace(0, 1, self.bins), repeat=self.n_observations)))

        return np.random.rand(self.num_rbf, self.n_observations)

    def __activate(self, _observation):
        inp_ = (_observation - self.lower) / self.amp  # Normalize
        diff = inp_ - self.centroids
        acts = np.exp(-0.5 * pow(np.linalg.norm(diff, axis=1), 2) / self.var)  # Activate RBFs
        return acts

    @staticmethod
    def __add_bias(x):
        return np.concatenate((x, [1.0]))

    def __learn(self, s_, r, done):
        if self.prev_observation is None:
            return 0

        s = self.prev_observation
        a = self.prev_action
        r = self.ep_reward

        acts = self.__activate(s)
        inp = self.__add_bias(acts)  # Add bias
        self.trace = inp + (self.decay * self.discount) * self.trace  # Update traces
        q = self.weights.dot(inp)  # Feed forward to linear output layer
        curr_q = q[a]

        acts = self.__activate(s_)
        new_inp = self.__add_bias(acts)

        if done:  # Last step
            target = r
        else:
            next_q = self.weights.dot(new_inp)
            max_q = next_q.max()
            target = r + self.discount * max_q
        td_error = target - curr_q

        delta = self.learning_rate * td_error * self.trace

        self.weights[a, :] += delta  # Update output layer for last action
        return td_error

    def __choose_action(self, observation):
        if np.random.uniform() < self.epsilon:  # Exploration
            return self.action_space.sample()

        acts = self.__activate(observation)
        inp = self.__add_bias(acts)  # Add bias
        q = self.weights.dot(inp)  # Feed forward to linear output layer
        return q.argmax()

    def act(self, observation, reward, done):

        self.ep_reward += reward

        self.__learn(observation, reward, done)

        action = self.__choose_action(observation)

        self.prev_observation = observation.copy()
        self.prev_action = action

        if done:
            self.episode_count += 1
            self.learning_rate = 1 / (self.episode_count + 1) ** 0.5
            self.epsilon = 1 / (self.episode_count + 1) ** 0.2
            self.ep_reward = 0
            self.prev_observation = None
            self.prev_action = None

        return action
