from rlagents import validate


class History:
    def __init__(self, size=100):
        validate.number_range(size, 0, 10000)

        self.size = size

        self.observations = []
        self.rewards = []
        self.done = []
        self.actions = []
        self.parameters = []

    def store(self, observation=None, reward=None, done=None, action=None, parameters=None):
        if observation is not None:
            self.observations.append(observation)

            if len(self.observations) > self.size:
                self.observations = self.observations[-self.size:]

        if reward is not None:
            self.rewards.append(reward)

            if len(self.rewards) > self.size:
                self.rewards = self.rewards[-self.size:]

        if done is not None:
            self.done.append(done)

            if len(self.done) > self.size:
                self.done = self.done[-self.size:]

        if action is not None:
            self.actions.append(action)

            if len(self.actions) > self.size:
                self.actions = self.actions[-self.size:]

        if parameters is not None:
            self.parameters.append(parameters)

            if len(self.parameters) > self.size:
                self.parameters = self.parameters[-self.size:]

    # Returns the last x histories
    def retrieve(self, last):
        validate.number_range(last, len(self.observations), self.size, min_eq=True, max_eq=True)

        if len(self.observations) == 0:
            return None

        last_o = min(last, len(self.observations))
        last_r = min(last, len(self.rewards))
        last_d = min(last, len(self.done))
        last_a = min(last, len(self.actions))
        last_p = min(last, len(self.parameters))

        return {"observations": self.observations[-last_o] if last_o > 0 else [],
                "rewards": self.rewards[-last_r] if last_r > 0 else [],
                "done": self.done[-last_d] if last_d > 0 else [],
                "actions": self.actions[-last_a] if last_a > 0 else [],
                "parameters": self.parameters[-last_p] if last_p > 0 else []}
