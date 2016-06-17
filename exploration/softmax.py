import numpy as np
from rlagents import validate


# Softmax selects the action to take based on the actions value relative to other actions
class Softmax:
    def __init__(self, temperature=0.1):
        validate.number_range(temperature, minimum=0)
        self.temperature = temperature

    def choose_action(self, model, observation):
        validate.model(model)

        q_s = model.action_value(observation)

        probabilities = []

        for action in range(len(q_s)):
            numerator = np.e ** (q_s[action]/self.temperature)
            denominator = sum([np.e ** (q_s[other]/self.temperature) for other in range(len(q_s))])
            chance = numerator / denominator

            probabilities.append(chance)

        choice = np.random.uniform()
        cum_sum = 0

        action = None

        for act, value in enumerate(probabilities):
            cum_sum += value

            if cum_sum >= choice:
                action = act
                break

        return action

    def update(self):
        pass
