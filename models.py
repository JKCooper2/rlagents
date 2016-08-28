import numpy as np


class ModelBase(object):
    def __init__(self, action_fa=None, observation_fa=None):
        self.observation_fa = observation_fa
        self.action_fa = action_fa

    @property
    def n_observations(self):
        return self.observation_fa.num_discrete

    @property
    def n_actions(self):
        return self.action_fa.num_discrete

    def action_value(self, observation):
        """Returns an array of values corresponding to possible actions in a state"""
        raise NotImplementedError

    def state_value(self, observation):
        """Returns the value for being in a particular state assuming the best action is taken"""
        raise NotImplementedError

    def state_action_value(self, observation, action):
        """Returns the expected value for performing an action in state"""
        raise NotImplementedError

    def export_values(self):
        raise NotImplementedError

    def import_values(self, values):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def update(self, observation, action, value):
        pass

    def configure(self, action_fa, observation_fa):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError


class DefaultModel(ModelBase):
    """Default Model where nothing happens"""
    def __init__(self, action_fa=None, observation_fa=None):
        ModelBase.__init__(self, action_fa, observation_fa)

        if self.action_fa is not None and self.observation_fa is not None:
            self.configure(self.action_fa, self.observation_fa)

    def configure(self, action_fa, observation_fa):
        self.action_fa = action_fa
        self.observation_fa = observation_fa

    def action_value(self, observation):
        return [0] * self.n_actions

    def state_value(self, observation):
        return [0]

    def state_action_value(self, observation, action):
        return [0]

    def export_values(self):
        return [0]

    def import_values(self, values):
        pass

    def reset(self):
        pass

    def update(self, observation, action, value):
        pass

    def export(self):
        return {"Type": "Default"}


class WeightedLinearModel(ModelBase):
    """
    Applies weighted linear function to an observation
    """
    def __init__(self, action_fa=None, observation_fa=None, bias=True, normalise=False):
        ModelBase.__init__(self, action_fa, observation_fa)

        self.bias = bias
        self.normalise = normalise

        self.weights = None
        self.bias_weight = None

        if self.action_fa is not None and self.observation_fa is not None:
            self.configure(self.action_fa, self.observation_fa)

    def configure(self, action_fa, observation_fa):
        self.action_fa = action_fa
        self.observation_fa = observation_fa
        self.reset()

    # Returns the action-value array
    def action_value(self, observation):
        observation = self.observation_fa.convert(observation)
        score = observation.dot(self.weights) + self.bias_weight
        return score[0]

    def state_value(self, observation):
        return max(self.action_value(observation))

    def state_action_value(self, observation, action):
        if not isinstance(action, int):
            raise TypeError("State Action Value current only accepts ints")

        return self.action_value(observation)[action]

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
        self.bias_weight = np.random.randn(self.n_actions).reshape(1, self.n_actions) if self.bias else np.zeros(self.n_actions).reshape(1, self.n_actions)

    def export(self):
        return {"Type": "Weighted Linear Model",
                "Bias": self.bias,
                "Normalise": self.normalise}

class TabularModel(ModelBase):
    def __init__(self, action_fa=None, observation_fa=None, mean=0.0, std=1.0):
        ModelBase.__init__(self, action_fa, observation_fa)

        self.mean = mean
        self.std = std

        self.weights = np.random.normal(self.mean, scale=self.std, size=(self.observation_fa.n_total, self.action_fa.n_total))
        self.keys = None

        if self.action_fa is not None and self.observation_fa is not None:
            self.configure(self.action_fa, self.observation_fa)

    def configure(self, action_fa, observation_fa):
        self.action_fa = action_fa
        self.observation_fa = observation_fa
        self.reset()

    def state_value(self, observation):
        observation = self.observation_fa.convert(observation)
        return max(self.weights[observation])

    def action_value(self, observation):
        observation = self.observation_fa.convert(observation)
        return self.weights[observation]

    def state_action_value(self, observation, action):
        return self.action_value(observation)[action]

    def update(self, observation, action, value):
        observation = self.observation_fa.convert(observation)
        self.weights[observation][action] = value

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

    def export(self):
        return {"Type": "Tabular Model",
                "Mean": self.mean,
                "Std": self.std}
