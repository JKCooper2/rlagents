class Value:
    def __init__(self, value_dict, memory, gamma=0.95, episodic=True):
        self.value_dict = value_dict
        self.memory = memory
        self.gamma = gamma
        self.episodic = episodic  # Update one per episode (True) or every step (False)

    def learn(self):
        if self.episodic and self.memory.last().done is False:
            return

        ep_starts = [0] + [key for key, val in self.memory.history.iteritems() if val.done]
        ep_starts = sorted(ep_starts)

        v = 0
        for t in reversed(xrange(ep_starts[-2], ep_starts[-1])):
            v = self.gamma * v + self.memory.history[t].reward

            self.value_dict[t] = v
