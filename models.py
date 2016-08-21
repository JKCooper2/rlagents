import numpy as np


class ModelBase(object):
    def __init__(self, action_fa, observation_fa):
        self.observation_fa = observation_fa
        self.action_fa = action_fa

    @property
    def n_observations(self):
        return self.observation_fa.num_discrete

    @property
    def n_actions(self):
        return self.action_fa.num_discrete

    def score(self, observation):
        raise NotImplementedError

    def export_values(self):
        raise NotImplementedError

    def import_values(self, values):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class WeightedLinearModel(ModelBase):
    """
    Applies weighted linear function to an observation
    """
    def __init__(self, action_fa, observation_fa, bias=True, normalise=False):
        ModelBase.__init__(self, action_fa, observation_fa)

        self.bias = bias
        self.normalise = normalise

        self.weights = np.random.randn(self.n_observations * self.n_actions).reshape(self.n_observations, self.n_actions)
        self.bias_weight = np.random.randn(self.n_actions).reshape(1, self.n_actions)

    def score(self, observation):
        observation = self.observation_fa.convert(observation)
        score = observation.dot(self.weights) + self.bias_weight
        return score[0]

    # Returns the action-value array
    def action_value(self, observation):
        observation = self.observation_fa.convert(observation)
        return self.action_fa.convert(self.score(observation))

    # Returns best action to perform along with it's value
    def action(self, observation):
        observation = self.observation_fa.convert(observation)
        return self.action_fa.convert(self.score(observation))

    def state_value(self, observation):
        observation = self.observation_fa.convert(observation)
        return max(self.score(observation))

    def export_values(self):
        values = np.concatenate((self.weights, self.bias_weight)) if self.bias else self.weights
        return values.flatten().copy()

    def import_values(self, weights):
        if len(weights) != self.n_observations * self.n_actions + (self.n_actions * int(self.bias)):
            raise ValueError("Value count can't be inserted into model")

        if self.normalise:
            weights /= np.linalg.norm(weights)

        self.weights = np.array(
            weights[:self.n_observations * self.n_actions].reshape(self.n_observations, self.n_actions))

        if self.bias:
            self.bias_weight = np.array(weights[self.n_observations * self.n_actions:].reshape(1, self.n_actions))

    def reset(self):
        self.weights = np.random.randn(self.n_observations * self.n_actions).reshape(self.n_observations, self.n_actions)
        self.bias_weight = np.random.randn(self.n_actions).reshape(1, self.n_actions)


class TabularModel(ModelBase):
    def __init__(self, action_fa, observation_fa, mean=0.0, std=1.0):
        ModelBase.__init__(self, action_fa, observation_fa)

        self.mean = mean
        self.std = std

        self.weights = np.random.normal(self.mean, scale=self.std, size=(self.observation_fa.n_total, self.action_fa.n_total))
        self.keys = None

        self.reset()

    def score(self, observation):
        pass

    def state_value(self, observation):
        observation = self.observation_fa.convert(observation)
        return max(self.weights[observation])

    def action_value(self, observation):
        observation = self.observation_fa.convert(observation)
        return self.weights[observation]

    def action(self, observation):
        observation = self.observation_fa.convert(observation)
        return np.argmax(self.weights[observation])

    def export_values(self):
        values = []

        for i in range(len(self.weights)):
            values.extend(self.weights[i])

        return np.array(values)

    def import_values(self, values):
        for i in range(len(values)/self.n_actions):
            self.weights[i] = np.array(values[self.n_actions * i: self.n_actions * i + self.n_actions])

    def reset(self):
        self.weights = np.random.normal(self.mean, scale=self.std, size=(self.observation_fa.n_total, self.action_fa.n_total))
        self.keys = None
