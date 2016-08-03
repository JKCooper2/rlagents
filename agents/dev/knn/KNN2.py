from rlagents.exploration.epsilon_greedy import EpsilonGreedy
from rlagents.functions.decay import FixedDecay
from rlagents.memory.memory import LongTerm
from rlagents.models.knn import KNN
from rlagents.optimisation.value import Value
import time


class EpisodicAgent:
    def __init__(self, action_space, observation_space, exploration=None, memory=None, model=None, optimiser=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = "KNN-Agent"

        self.exploration = exploration if exploration is not None else EpsilonGreedy(action_space, decay=FixedDecay(0.5, 0.995, 0.05))
        self.memory = memory if memory is not None else LongTerm(size=250)
        self.model = model if model is not None else KNN(self.memory, neighbours=10)
        self.optimiser = optimiser if optimiser is not None else Value(self.model.values, self.memory, gamma=1)
        self.timer = [0, 0, 0]


    def act(self, observation, reward, done):

        v1 = time.time()
        action = self.exploration.choose_action(self.model, observation)
        self.timer[0] += time.time() - v1

        v1 = time.time()
        self.memory.store(observation, action, reward, done)
        self.timer[1] += time.time() - v1
        v1 = time.time()
        self.optimiser.learn(observation, reward, done)
        self.timer[2] += time.time() - v1

        if done:
            self.exploration.update()
            print round(self.timer[0] * 100 / sum(self.timer), 4), round(self.timer[1] * 100 / sum(self.timer), 4), round(self.timer[2] * 100 / sum(self.timer), 4), sum(self.timer)
            self.timer = [0, 0, 0]

        return action
