import numpy as np


class KNN:
    def __init__(self, memory, neighbours=100):
        self.memory = memory
        self.neighbours = neighbours
        self.values = {}

        self.timer = [0, 0, 0, 0]

    def action(self, observation):
        action_values = self.action_value(observation)

        if action_values is None:
            return None

        return action_values[0][0]

    def action_value(self, observation):
        if len(self.values) > self.neighbours:
            last_ep = self.memory.retrieve_last(1)
            steps = int(last_ep.iloc[0]['Step'])
            obs_history = self.memory.retrieve_last(self.memory.size).iloc[:-steps]

            # exploit: find the few closest states and pick the action that led to highest rewards
            # 1. find k nearest neighbors
            if isinstance(observation, (int, np.int64)):
                ds = (obs_history.iloc[:, 5:] - observation) ** 2

            else:
                ds = np.sum((obs_history.iloc[:, 5:] - observation) ** 2, axis=1)  # L2 distance

            ix = np.argsort(ds)  # sorts ascending by distance
            ix = ix[:self.neighbours]  # crop to only some number of nearest neighbors

            obs_index = obs_history.index.values
            actions = obs_history.loc[obs_index[ix]]['Action']

            adict = {}
            ndict = {}
            for i in ix:
                vv = self.values[obs_index[i]]
                aa = actions[i]
                adict[aa] = adict.get(aa, 0) + vv
                ndict[aa] = ndict.get(aa, 0) + 1

            for a in adict:  # normalize by counts
                adict[a] = adict[a] / ndict[a]

            its = [(y, x) for x, y in adict.iteritems()]
            its.sort(reverse=True)  # descending

            return [(y, x) for x, y in its]

        return None
