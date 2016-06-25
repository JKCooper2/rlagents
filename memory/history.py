class History:
    def __init__(self, key=None, observation=None, action=None, reward=None, done=None, parameters=None):
        self.key = key
        self.observation = observation
        self.action = action
        self.reward = reward
        self.done = done
        self.parameters = parameters

    def report(self):
        print "Key:", self.key
        print "Observation:", self.observation
        print "Action:", self.action
        print "Reward:", self.reward
        print "Done:", self.done
        print "Parameters:", self.parameters
