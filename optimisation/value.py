class Value:
    def __init__(self, value_dict, memory, gamma=0.95):
        self.value_dict = value_dict
        self.memory = memory
        self.gamma = gamma

    def learn(self, observation, reward, done):
        if not done:
            return

        last_ep = self.memory.retrieve_last(1)

        if last_ep is None:
            return

        steps = int(last_ep.iloc[0]['Step'])
        ep_steps = self.memory.retrieve_last(steps)

        index = ep_steps.index.values

        # Include this episode (final episode)
        v = reward
        self.value_dict[index[-1] + 1] = v

        for t in reversed(index):
            v = self.gamma * v + ep_steps.loc[t]['Reward']
            self.value_dict[t] = v
