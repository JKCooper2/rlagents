import numpy as np

from rlagents.exploration.epsilon_greedy import EpsilonGreedy
from rlagents.functions.decay import FixedDecay
from rlagents.memory.memory import LongTerm
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
    def __init__(self, action_space, observation_space, exploration=None, memory=None, model=None, optimiser=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = "KNN-Agent"

        self.exploration = exploration if exploration is not None else EpsilonGreedy(action_space, decay=FixedDecay(0.5, 0.995, 0.05))
        self.memory = memory if memory is not None else LongTerm(size=5000)
        self.model = model if model is not None else KNN(self.memory, neighbours=50)
        self.optimiser = optimiser if optimiser is not None else Value(self.model.values, self.memory)

    def act(self, observation, reward, done):
        reward += -0.01

        self.memory.store(reward=reward, done=done, step=True)

        self.optimiser.learn()

        action = self.exploration.choose_action(self.model, observation)

        if done:
            self.exploration.update()

        self.memory.store(observation=observation, action=action)

        return action
