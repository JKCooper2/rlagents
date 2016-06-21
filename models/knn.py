import numpy as np


class KNN:
    def __init__(self, history, neighbours=100):
        self.history = history
        self.neighbours = neighbours

        self.values = {}

    def action(self, observation):
        if len(self.values) > self.neighbours:

            # exploit: find the few closest states and pick the action that led to highest rewards
            # 1. find k nearest neighbors
            ds = np.sum((self.history.observations[:len(self.values)] - observation) ** 2, axis=1)  # L2 distance
            ix = np.argsort(ds)  # sorts ascending by distance
            ix = ix[:self.neighbours]  # crop to only some number of nearest neighbors

            # find the action that leads to most success. do a vote among actions
            adict = {}
            ndict = {}
            for i in ix:
                vv = self.values[i]
                aa = self.history.actions[i]
                vnew = adict.get(aa, 0) + vv
                adict[aa] = vnew
                ndict[aa] = ndict.get(aa, 0) + 1

            for a in adict:  # normalize by counts
                adict[a] = adict[a] / ndict[a]

            its = [(y, x) for x, y in adict.iteritems()]
            its.sort(reverse=True)  # descending
            action = its[0][1]

        else:
            action = None

        return action

    def action_value(self):
        pass