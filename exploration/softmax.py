import numpy as np


class Softmax:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def choose_action(self, model, observation):
        q_s = model.action_value(observation)

        probabilities = []

        for action in range(len(q_s)):
            numerator = np.e ** (q_s[action]/self.temperature)
            denominator = sum([np.e ** (q_s[other]/self.temperature) for other in range(len(q_s))])
            chance = numerator / denominator

            probabilities.append(chance)

        choice = np.random.uniform()
        cum_sum = 0

        for action, value in enumerate(probabilities):
            cum_sum += value

            if cum_sum >= choice:
                return action

        print "ERROR CHOOSING WITH SOFTMAX", probabilities, choice, cum_sum

    def update(self):
        pass