import pandas as pd


class LongTerm:
    def __init__(self, size=100):
        self.max_size = size

        self.memory = None
        self.index = 0
        self.episode = 1
        self.step = 1

    @property
    def size(self):
        return len(self.memory) if self.memory is not None else 0

    def store(self, observation, action, reward, done):
        # Lazy initialisation so column names are correct
        if self.memory is None:
            columns = ["Episode", "Step", "Action", "Reward", "Done"] + ["O" + str(i) for i in range(len(observation))]
            self.memory = pd.DataFrame(columns=columns)

        # Apply the reward backwards
        if self.size > 0:
            self.memory.loc[self.index - 1, "Reward"] = reward  # Apply the reward backwards so it's correct

 # Need to speed this up
        self.memory.loc[self.index] = [self.episode, self.step, action, None, done] + list(observation)

        # If over size remove earliest episode
        if self.size > self.max_size:
            # Could be faster as well by knowing how many steps in the first ep and just dropping those rows
            ep_num = self.memory.iloc[0]['Episode']
            self.memory = self.memory[self.memory["Episode"] != ep_num]

        self.step += 1
        self.index += 1

        if done:
            self.episode += 1
            self.step = 1

    # Returns the last x histories
    def retrieve_last(self, last):
        last = min(last, self.size)

        if self.size < 2:
            return None

        return self.memory[-last-1:-1]  # Only return where reward is known
