import numpy as np

from rlagents.exploration.epsilon_greedy import EpsilonGreedy
from rlagents.functions.decay import FixedDecay
from rlagents.history.history import History
from rlagents.models.knn import KNN
from rlagents.optimisation.value import Value

class EpisodicAgent:
    """
    Episodic agent is a simple nearest-neighbor based agent:
    - At training time it remembers all tuples of (state, action, reward).
    - After each episode it computes the empirical value function based
        on the recorded rewards in the episode.
    - At test time it looks up k-nearest neighbors in the state space
        and takes the action that most often leads to highest average value.
    """
    def __init__(self, action_space, observation_space, exploration=None, history=None, model=None, optimiser=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = "KNN-Agent"

        self.exploration = exploration if exploration is not None else EpsilonGreedy(action_space, decay=FixedDecay(1.0, 0.98, 0))
        self.history = history if history is not None else History(size=50000)
        self.model = KNN(self.history, neighbours=500)
        self.optimiser = Value(self.model.values, self.history)

    def act(self, observation, reward, done):

        self.optimiser.learn()

        action = self.exploration.choose_action(self.model, observation)

        self.history.store(observation, reward, done, action)

        if done:  # episode Ended;
            # decay exploration probability
            self.exploration.update()

        return action
