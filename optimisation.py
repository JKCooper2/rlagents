import warnings

from rlagents.functions.decay import DecayBase, FixedDecay


class OptimiserBase(object):
    def __init__(self, model=None, memory=None):
        self.model = model
        self.memory = memory

    def _is_valid(self):
        return self.model is not None and self.memory is not None

    def run(self):
        raise NotImplementedError

    def configure(self, model, memory):
        self.model = model
        self.memory = memory

    def export(self):
        raise NotImplementedError


class DefaultOptimiser(OptimiserBase):
    def run(self):
        pass

    def export(self):
        return {"Type": "Default"}


class TemporalDifference(OptimiserBase):
    def __init__(self, model=None, memory=None, discount=0.95, learning_rate=None):
        OptimiserBase.__init__(self, model, memory)
        self.discount = discount
        self.learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        if not isinstance(lr, DecayBase):
            lr = FixedDecay(1, decay=0.995, minimum=0.05)
            warnings.warn('Learning Rate type invalid, using default. ({0})'.format(lr))

        self._learning_rate = lr

    def run(self):
        assert self._is_valid()

        if self.memory.count('observations') < 2:
            return

        m = self.memory.fetch_last(2)

        observation = m['observations'][1]
        done = m['done'][1]
        reward = m['rewards'][1]

        prev_obs = m['observations'][0]
        prev_action = m['actions'][0]

        future = self.model.state_value(observation) if not done else 0.0

        new_val = self.model.state_action_value(prev_obs, prev_action) + self.learning_rate.value * (reward + self.discount * future - self.model.state_action_value(prev_obs, prev_action))

        self.model.update(prev_obs, prev_action, new_val)

        if done:
            self.learning_rate.update()

    def export(self):
        return {"Type": "Temporal Difference",
                "Discount": self.discount,
                "Learning Rate": self.learning_rate.export()}

