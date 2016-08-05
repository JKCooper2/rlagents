import warnings
import copy
import numpy as np

from gym.spaces import tuple_space, box, discrete

from rlagents.functions.decay import DecayBase, FixedDecay
from rlagents.exploration import EpsilonGreedy, ExplorationBase
from rlagents.function_approximation import DiscreteFA, SingleTiling
from rlagents.models import ModelBase, TabularModel
from rlagents.history import History
from rlagents.optimisation.evolutionary import HillClimbing, EvolutionaryBase


class AgentBase(object):
    def __init__(self, action_space, observation_space, name=None, alg_id=None):
        self.name = name
        self.alg_id = alg_id
        self.action_space = action_space
        self.observation_space = observation_space

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, a_s):
        self._action_space = a_s

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, o_s):
        self._observation_space = o_s

    def act(self, observation, reward, done):
        raise NotImplementedError


class StandardAgent(AgentBase):
    def __init__(self, action_space, observation_space, discount=0.95, learning_rate=None, exploration=None, observation_fa=None, history=None, model=None):
        AgentBase.__init__(self, action_space, observation_space, name="Standard Agent", alg_id="alg_OwSFZtRR2eZYkcxkG74Q")

        self.discount = discount
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.observation_fa = observation_fa
        self.history = history

        self.model = model

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        if not isinstance(lr, DecayBase):
            lr = FixedDecay(1, decay=0.995, minimum=0.05)
            warnings.warn('Learning Rate type invalid, using default. ({0})'.format(lr))

        self._learning_rate = lr

    @property
    def exploration(self):
        return self._exploration

    @exploration.setter
    def exploration(self, ex):
        if not isinstance(ex, ExplorationBase):
            ex = EpsilonGreedy(self.action_space, decay=FixedDecay(1, 0.997, 0.05))
            warnings.warn('Exploration type invalid, using default. ({0})'.format(ex))

        self._exploration = ex

    @property
    def observation_fa(self):
        return self._observation_fa

    @observation_fa.setter
    def observation_fa(self, ofa):
        if ofa is None:
            if isinstance(self.observation_space, tuple_space.Tuple):
                ofa = DiscreteFA([space.n for space in self.observation_space.spaces])

            elif isinstance(self.observation_space, box.Box):
                ofa = SingleTiling(self.observation_space, 8)

            elif isinstance(self.observation_space, discrete.Discrete):
                ofa = DiscreteFA([self.observation_space.n])

        self._observation_fa = ofa

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        if not isinstance(m, ModelBase):
            m = TabularModel(self.action_space.n, self.observation_fa)
            warnings.warn("Model type invalid, using defaults")

        self._model = m

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, h):
        if h is None:
            h = History(size=1)

        self._history = h

    def __choose_action(self, observation):
        return self.exploration.choose_action(self.model, observation)

    def __learn(self, observation_key, reward, done):
        last_turn = self.history.retrieve(1)

        if last_turn is None:
            return

        prev_obs = last_turn["observations"]
        prev_action = last_turn["actions"]

        future = self.model.state_value(observation_key) if not done else 0.0
        self.model.weights[prev_obs][prev_action] += self.learning_rate.value * (reward + self.discount * future - self.model.weights[prev_obs][prev_action])

    def act(self, observation, reward, done):
        observation_key = self.observation_fa.convert(observation)

        self.__learn(observation_key, reward, done)
        action = self.__choose_action(observation_key)

        self.history.store(observation=observation_key, action=action)

        if done:
            self.exploration.update()
            self.learning_rate.update()

        return action


class EvolutionaryAgent(AgentBase):
    def __init__(self, action_space, observation_space, model=None, evolution=None, batch_size=1):
        AgentBase.__init__(self, action_space, observation_space, name="Evolutionary Agent")

        self.evolution = evolution
        self.episode_reward = 0

        self.model = model

        self.batch_size = batch_size  # Number of samples run per batch
        self.batch_test = 0
        self.batch_results = []
        self.batch = self.__generate_batch()

    @property
    def evolution(self):
        return self._evolution

    @evolution.setter
    def evolution(self, e):
        if not isinstance(e, EvolutionaryBase):
            e = HillClimbing
            warnings.warn("Evolution not subclass of EvolutionaryBase, using default HillClimbing")

        self._evolution = e

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        if not isinstance(m, ModelBase):
            raise TypeError("Model not a valid ModelBase")

        self._model = m

    def __generate_batch(self):
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


class RandomAgent(AgentBase):
    def __init__(self, action_space, observation_space):
        AgentBase.__init__(self, action_space, observation_space, name="Random Agent", alg_id = "alg_MhPaN5c4TJOFS4tVFh8x3A")

    def act(self, observation, reward, done):
        return self.__validate_action(self.action_space.sample())

    def __validate_action(self, action):
        if hasattr(action, '__iter__'):
            for i in range(len(action)):
                self.__validate_action(action[i])
        elif np.isnan(action):
            action = np.random.normal(0, 1.0)

        return action
