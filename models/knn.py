import numpy as np


class KNN:
    def __init__(self, memory, neighbours=100):
        self.memory = memory
        self.neighbours = neighbours
        self.values = {}

    def action(self, observation):
        action_values = self.action_value(observation)

        if action_values is None:
            return None

        return action_values[0][0]

    def action_value(self, observation):
        if len(self.values) > self.neighbours:
            history = [valid for valid in self.memory.history.values() if
                       valid.key in self.values and valid.observation is not None]

            obs_history = np.array([item.observation for item in history])

            # exploit: find the few closest states and pick the action that led to highest rewards
            # 1. find k nearest neighbors
            if isinstance(observation, (int, np.int64)):
                ds = (obs_history - observation) ** 2

            else:
                ds = np.sum((obs_history - observation) ** 2, axis=1)  # L2 distance

            ix = np.argsort(ds)  # sorts ascending by distance
            ix = ix[:self.neighbours]  # crop to only some number of nearest neighbors

            # find the action that leads to most success. do a vote among actions
            adict = {}
            ndict = {}
            for i in ix:
                key = history[i].key
                vv = self.values[key]
                aa = self.memory.history[key].action
                vnew = adict.get(aa, 0) + vv
                adict[aa] = vnew
                ndict[aa] = ndict.get(aa, 0) + 1

            for a in adict:  # normalize by counts
                adict[a] = adict[a] / ndict[a]

            its = [(y, x) for x, y in adict.iteritems()]
            its.sort(reverse=True)  # descending

            return [(y, x) for x, y in its]

        return None
