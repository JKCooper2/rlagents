from rlagents import validate
from rlagents.memory.history import History
import numpy as np


class LongTerm:
    def __init__(self, size=100, forget='oldest'):
        validate.number_range(size, 0, 100000)

        self.max_size = size
        self.history = {}
        self.key = 0  # Memory value

        if forget not in ['oldest', 'newest', 'random']:
            raise ValueError("Forget strategy is invalid")

        self.forget = forget

    @property
    def size(self):
        return len(self.history)

    def __forget_key(self, memory):
        if self.forget == 'oldest':
            return min(memory)

        if self.forget == 'newest':
            return max(memory)

        if self.forget == 'random':
            return np.random.choice(memory.keys())

    def store(self, observation=None, reward=None, done=None, action=None, parameters=None, step=False, shift=0):
        if (self.key + shift) not in self.history:
            self.history[self.key + shift] = History(key=self.key+shift)

        if observation is not None:
            self.history[self.key + shift].observation = observation

        if reward is not None:
            self.history[self.key + shift].reward = reward

        if done is not None:
            self.history[self.key + shift].done = done

        if action is not None:
            self.history[self.key + shift].action = action

        if parameters is not None:
            self.history[self.key + shift].parameters = parameters

        if self.size > self.max_size:
            self.history.pop(self.__forget_key(self.history), None)

        # Update key if step is set to true
        if step:
            self.key += 1

    # Returns the last x histories
    def retrieve_x(self, last):
        validate.number_range(last, 1, self.size, min_eq=True, max_eq=True)

        return {key: self.history[key] for key in reversed(sorted(self.history.keys()[-last:]))}

    def last(self):
        if self.size == 0:
            return History()

        return self.history[max(self.history.iterkeys())]
