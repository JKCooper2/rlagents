class Value:
    def __init__(self, value_dict, history, gamma=0.95, episodic=True):
        self.value_dict = value_dict
        self.history = history
        self.gamma = gamma
        self.episodic = episodic  # Update one per episode (True) or every step (False)

    def learn(self):
        if self.episodic and (len(self.history.done) == 0 or self.history.done[-1] == False):
            return

        ep_starts = [0] + [loc for loc, val in enumerate(self.history.done) if val]

        v = 0
        for t in reversed(xrange(ep_starts[-2], ep_starts[-1])):
            v = self.gamma * v + self.history.rewards[t]
            self.value_dict[t] = v