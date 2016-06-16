import copy
from rlagents.exploration.epsilon_greedy import EpsilonGreedy
from rlagents.models.linear import BinaryLinearModel
from rlagents.optimisation.evolutionary.cross_entropy import CrossEntropy
from rlagents.optimisation.evolutionary.simulated_annealing import SimulatedAnnealing


class EvolutionaryAgent:
    def __init__(self, action_space, observation_space, exploration=None, model=None, evolution=None, batch_size=1):
        self.name = "EvolutionaryAgent"
        self.action_space = action_space
        self.observation_space = observation_space

        self.batch_size = batch_size  # Number of samples run per batch

        self.batch_test = 0
        self.batch_results = []

        self.episode_reward = 0

        self.exploration = self.__set_exploration(action_space, exploration)
        self.model = self.__set_model(model, action_space, observation_space)
        self.evolution = self.__set_evolution(evolution)

        self.batch = self.__set_batch()

        self.__validate_setup()

    def __validate_setup(self):
        assert hasattr(self.exploration, 'choose_action') and callable(getattr(self.exploration, 'choose_action'))
        assert hasattr(self.exploration, 'update') and callable(getattr(self.exploration, 'update'))

    @staticmethod
    def __set_exploration(action_space, exploration):
        if exploration is None:
            return EpsilonGreedy(action_space, epsilon=0, decay=0, minimum=0)

        return exploration

    @staticmethod
    def __set_model(model, action_space, observation_space):
        if model is None:
            return BinaryLinearModel(len(observation_space.low), bias=True)

        return model

    @staticmethod
    def __set_evolution(evolution):
        if evolution is None:
            return SimulatedAnnealing()

        return evolution

    def __set_batch(self):
        batch = [copy.copy(self.model) for _ in range(self.batch_size)]
        for i in range(len(batch)):
            batch[i].reset()

        return batch

    def __choose_action(self, observation):
        score = self.batch[self.batch_test].score(observation)
        obs = [i == score for i in range(self.action_space.n)]

        return self.exploration.choose_action(obs)

    def act(self, observation, reward, done):
        action = self.__choose_action(observation)

        self.episode_reward += reward

        if done:
            self.batch_results.append(self.episode_reward)
            self.episode_reward = 0
            self.batch_test += 1

            # If all members of current generation have been tested
            if self.batch_test == self.batch_size:
                self.batch = self.evolution.next_generation(self.batch, self.batch_results)
                self.batch_test = 0
                self.batch_results = []

        return action