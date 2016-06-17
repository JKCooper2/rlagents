import copy
from rlagents.models.linear import DiscreteActionLinearModel
from rlagents.optimisation.evolutionary.hill_climbing import HillClimbing


class EvolutionaryAgent:
    def __init__(self, action_space, observation_space, model=None, evolution=None, batch_size=1):
        self.name = "EvolutionaryAgent"
        self.action_space = action_space
        self.observation_space = observation_space

        self.batch_size = batch_size  # Number of samples run per batch

        self.batch_test = 0
        self.batch_results = []

        self.episode_reward = 0

        self.model = self.__set_model(model, action_space, observation_space)
        self.evolution = self.__set_evolution(evolution)

        self.batch = self.__set_batch()

    @staticmethod
    def __set_model(model, action_space, observation_space):
        if model is None:
            return DiscreteActionLinearModel(action_space, observation_space)

        return model

    @staticmethod
    def __set_evolution(evolution):
        if evolution is None:
            return HillClimbing()

        return evolution

    def __set_batch(self):
        batch = [copy.copy(self.model) for _ in range(self.batch_size)]
        for i in range(len(batch)):
            batch[i].reset()

        return batch

    def __choose_action(self, observation):
        return self.batch[self.batch_test].action(observation)

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
