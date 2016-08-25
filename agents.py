import warnings
import copy
import numpy as np

from rlagents.functions.decay import FixedDecay
from rlagents.exploration import EpsilonGreedy, ExplorationBase
from rlagents.function_approximation import DefaultFA, SingleTiling, FunctionApproximationBase, DiscreteMaxFA
from rlagents.models import ModelBase, TabularModel
from rlagents.memory import ListMemory
from rlagents.optimisation import OptimiserBase
from rlagents.optimisation.exploratory import MonteCarlo
from rlagents.optimisation.evolutionary import HillClimbing, EvolutionaryBase


class AgentBase(object):
    def __init__(self, action_space, observation_space, action_fa, model):
        self.action_space = action_space
        self.observation_space = observation_space

        self.action_fa = action_fa
        self.model = model

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

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        if not isinstance(m, ModelBase):
            m = TabularModel(DefaultFA(self.action_space), SingleTiling(self.observation_space, 8))
            warnings.warn("Model type invalid, using defaults")

        self._model = m

    @property
    def action_fa(self):
        return self._action_fa

    @action_fa.setter
    def action_fa(self, afa):
        if not isinstance(afa, FunctionApproximationBase):
            afa = DiscreteMaxFA(self.action_space)  # For now as most envs testing are discrete
            warnings.warn("action_fa must inherit from FunctionApproximationBase using defaults")

        self._action_fa = afa

    def act(self, observation, reward, done):
        raise NotImplementedError


class ExploratoryAgent(AgentBase):
    def __init__(self, action_space, observation_space, action_fa=None, model=None, exploration=None, memory=None, optimiser=None):
        AgentBase.__init__(self, action_space, observation_space, action_fa, model)

        self.memory = memory
        self.action_fa = action_fa

        self.exploration = exploration
        self.optimiser = optimiser

    @property
    def exploration(self):
        return self._exploration

    @exploration.setter
    def exploration(self, ex):
        if not isinstance(ex, ExplorationBase):
            ex = EpsilonGreedy(self.action_space, decay=FixedDecay(1, 0.997, 0.05))
            warnings.warn('Exploration type invalid, using default. ({0})'.format(ex))

        ex.model = self.model
        self._exploration = ex

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, m):
        if m is None:
            m = ListMemory(size=2)
            m.new(['observations',
                   'actions',
                   'done',
                   'rewards',
                   'parameters'])

        self._memory = m

    @property
    def optimiser(self):
        return self._optimiser

    @optimiser.setter
    def optimiser(self, o):
        if not isinstance(o, OptimiserBase):
            o = MonteCarlo()
            # raise TypeError("Optimiser is not a valid OptimiserBase")

        self._optimiser = o

        # Probably not the best idea to require ordering of variable loading but simpler than
        # passing through variables each time
        self._optimiser.model = self.model
        self._optimiser.memory = self.memory

    def act(self, observation, reward, done):
        self.memory.store({'observations': observation, 'done': done, 'rewards': reward})

        self.optimiser.run()

        action_values = self.exploration.bias_action_value(observation)
        action = self.action_fa.convert(action_values)

        self.memory.store({'actions': action})

        if done:
            self.exploration.update()

        return action


class EvolutionaryAgent(AgentBase):
    """
    Parameters:
        batch_size - Number of models used in each batch
        times_run - Number of times the model values will be run. Essentially 1-indexed repeats
    """
    def __init__(self, action_space, observation_space, action_fa=None, model=None, evolution=None, batch_size=1, times_run=1):
        AgentBase.__init__(self, action_space, observation_space, action_fa, model)

        self.evolution = evolution
        self.episode_reward = 0

        self.times_run = times_run

        self.batch_size = batch_size  # Number of samples run per batch
        self.batch_test = 0
        self.batch_results = []
        self.batch = []

    @property
    def evolution(self):
        return self._evolution

    @evolution.setter
    def evolution(self, e):
        if not isinstance(e, EvolutionaryBase):
            e = HillClimbing()
            warnings.warn("Evolution not subclass of EvolutionaryBase, using default HillClimbing")

        self._evolution = e

    @property
    def times_run(self):
        return self._times_run

    @times_run.setter
    def times_run(self, tr):
        if not isinstance(tr, int) or tr < 1:
            raise ValueError("Times must be an int >= 1")

        self._times_run = tr

    def __generate_batch(self):
        batch = [copy.copy(self.model) for _ in range(self.batch_size)]
        for i in range(len(batch)):
            batch[i].reset()

        # Expands the batch by the required number of times run (order is important)
        expanded_batch = [b for b in batch for _ in range(self.times_run)]

        return expanded_batch

    def __compress_batch(self):
        """
        Combines batches and results together when repeats are used
        """
        # If no repeats then compress is unnecessary
        if self.times_run == 1:
            return self.batch, self.batch_results

        batch = []
        batch_results = []

        for i in range(self.batch_size):
            index = i * self.times_run
            batch.append(self.batch[index])
            batch_results.append(float(sum(self.batch_results[index:index + self.times_run]))/self.times_run)

        return batch, batch_results

    def act(self, observation, reward, done):
        # Allows batch_size to be set outside of init
        if not self.batch:
            self.batch = self.__generate_batch()

        action_values = self.batch[self.batch_test].action_value(observation)
        action = self.action_fa.convert(action_values)

        self.episode_reward += reward

        if done:
            self.batch_results.append(self.episode_reward)
            self.episode_reward = 0
            self.batch_test += 1

            # If all members of current generation have been tested
            if self.batch_test == self.batch_size * self.times_run:
                next_batch = self.evolution.next_generation(*self.__compress_batch())
                self.batch = [b for b in next_batch for _ in range(self.times_run)]
                self.batch_test = 0
                self.batch_results = []

        return action


class RandomAgent(AgentBase):
    def __init__(self, action_space, observation_space):
        action_fa = DefaultFA(action_space)
        observation_fa = DefaultFA(observation_space)
        AgentBase.__init__(self, action_space, observation_space, action_fa=action_fa, model=ModelBase(action_fa, observation_fa))

    def act(self, observation, reward, done):
        return self.__validate_action(self.action_space.sample())

    def __validate_action(self, action):
        if hasattr(action, '__iter__'):
            for i in range(len(action)):
                self.__validate_action(action[i])
        elif np.isnan(action):
            action = np.random.normal(0, 1.0)

        return action
