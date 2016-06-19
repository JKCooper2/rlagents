import numpy as np


class DiscreteActionLinearModel:
    def __init__(self, action_space, observation_space, bias=True, normalise=False):
        self.observation_space = observation_space
        self.n_observations = observation_space.shape[0]
        self.action_space = action_space
        self.n_actions = action_space.n
        self.bias = bias
        self.normalise = normalise

        self.weights = np.random.randn(self.n_observations * self.n_actions).reshape(self.n_observations, self.n_actions)
        self.bias_weight = np.random.randn(self.n_actions).reshape(1, self.n_actions)

    def __score(self, observation):
        return observation.dot(self.weights) + self.bias_weight

    # Returns the state value
    def state_value(self, observation):
        return max(self.__score(observation))

    # Returns the action-value array
    def action_value(self, observation):
        return self.__score(observation)

    # Returns the best action
    def action(self, observation):
        y = self.__score(observation)
        return y.argmax()

    def export_values(self):
        values = np.concatenate((self.weights, self.bias_weight)) if self.bias else self.weights
        return values.flatten().copy()

    def import_values(self, weights):
        if len(weights) != self.n_observations * self.n_actions + (self.n_actions * int(self.bias)):
            raise ValueError("Value count can't be inserted into model")

        if self.normalise:
            weights /= np.linalg.norm(weights)

        self.weights = np.array(weights[:self.n_observations * self.n_actions].reshape(self.n_observations, self.n_actions))

        if self.bias:
            self.bias_weight = np.array(weights[self.n_observations * self.n_actions:].reshape(1, self.n_actions))

    def reset(self):
        self.weights = np.random.randn(self.n_observations * self.n_actions).reshape(self.n_observations, self.n_actions)
        self.bias_weight = np.random.randn(self.n_actions).reshape(1, self.n_actions)


class ContinuousActionLinearModel:
    def __init__(self, action_space, observation_space, bias=True, normalise=False):
        self.observation_space = observation_space
        self.n_observations = observation_space.shape[0] if hasattr(observation_space, 'shape') else observation_space.n
        self.action_space = action_space
        self.n_actions = action_space.shape[0]
        self.bias = bias
        self.normalise = normalise

        self.weights = np.random.randn(self.n_observations * self.n_actions).reshape(self.n_observations, self.n_actions)
        self.bias_weight = np.random.randn(self.n_actions)

    def __score(self, observation):
        if hasattr(observation, 'dot'):
            return observation.dot(self.weights) + self.bias_weight

        return observation * self.weights + self.bias_weight

    # Returns the state value
    def state_value(self, observation):
        return max(self.__score(observation))

    # Returns the action-value array
    def action_value(self, observation):
        return self.__score(observation)

    # Returns the best action
    def action(self, observation):
        action = self.__score(observation)
        return np.clip(action[0], self.action_space.low, self.action_space.high)

    def export_values(self):
        values = np.concatenate((self.weights, self.bias_weight)) if self.bias else self.weights
        return values.flatten()

    def import_values(self, values):
        if len(values) != self.n_observations * self.n_actions + self.n_actions:
            raise ValueError("Value count can't be inserted into model")

        if self.normalise:
            values /= np.linalg.norm(values)

        self.weights = np.array(values[:self.n_observations * self.n_actions].reshape(self.n_observations, self.n_actions))

        if self.bias:
            self.bias_weight = np.array(values[self.n_observations * self.n_actions:].reshape(1, self.n_actions))

    def reset(self):
        self.weights = np.random.randn(self.n_observations * self.n_actions).reshape(self.n_observations, self.n_actions)
        self.bias_weight = np.random.randn(self.n_actions).reshape(1, self.n_actions)