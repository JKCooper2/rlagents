import warnings

from rlagents.function_approximation import DefaultFA, FunctionApproximationBase
from rlagents.models import ModelBase, DefaultModel
from rlagents.memory import MemoryBase, ListMemory
from rlagents.optimisation import DefaultOptimiser, OptimiserBase
from rlagents.exploration import DefaultExploration, ExplorationBase


class Agent(object):
    def __init__(self, action_space=None, observation_space=None, action_fa=None, observation_fa=None, model=None, exploration=None, memory=None, optimiser=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = memory

        self.action_fa = action_fa
        self.observation_fa = observation_fa
        self.model = model
        self.exploration = exploration
        self.optimiser = optimiser

        # Used by pool to understand what happened with an agent
        self.episode_reward = 0
        self.done = False

        self.configured = False

        if self.action_space is not None and self.observation_space is not None:
            self.configure(self.action_space, self.observation_space)

    def configure(self, action_space=None, observation_space=None, overwrite=False):
        if self.configured and overwrite is False:
            return

        if action_space is None and observation_space is None:
            return

        if action_space is not None:
            self.action_space = action_space
            self.action_fa.configure(self.action_space)

        if observation_space is not None:
            self.observation_space = observation_space
            self.observation_fa.configure(self.observation_space)

        self.model.configure(self.action_fa, self.observation_fa)
        self.exploration.configure(self.model)
        self.optimiser.configure(self.model, self.memory)
        self.configured = True

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        if not isinstance(m, ModelBase):
            m = DefaultModel()
            warnings.warn("Model type invalid, using defaults")

        self._model = m

    @property
    def action_fa(self):
        return self._action_fa

    @action_fa.setter
    def action_fa(self, afa):
        if not isinstance(afa, FunctionApproximationBase):
            afa = DefaultFA()
            warnings.warn("action_fa must inherit from FunctionApproximationBase using defaults")

        self._action_fa = afa

    @property
    def observation_fa(self):
        return self._observation_fa

    @observation_fa.setter
    def observation_fa(self, ofa):
        if not isinstance(ofa, FunctionApproximationBase):
            ofa = DefaultFA()
            warnings.warn("observation_fa must inherit from FunctionApproximationBase using defaults")

        self._observation_fa = ofa

    @property
    def exploration(self):
        return self._exploration

    @exploration.setter
    def exploration(self, ex):
        if not isinstance(ex, ExplorationBase):
            ex = DefaultExploration()
            warnings.warn('Exploration type invalid, using default. ({0})'.format(ex))

        self._exploration = ex

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, m):
        if not isinstance(m, MemoryBase):
            m = ListMemory(size=2)
            m.new(['observations',
                   'actions',
                   'done',
                   'rewards'])
            warnings.warn('Memory type invalid, using List. ({0})'.format(m))

        self._memory = m

    @property
    def optimiser(self):
        return self._optimiser

    @optimiser.setter
    def optimiser(self, o):
        if not isinstance(o, OptimiserBase):
            o = DefaultOptimiser()
            warnings.warn("Optimiser is not a valid OptimiserBase")

        self._optimiser = o

    def next_agent(self):
        """Keeps functionality in EnvManager consistent with Pools"""
        self.episode_reward = 0
        return self

    def act(self, observation, reward, done):
        if not self.configured:
            raise AssertionError("Agent must have run .configure() before taking an action")

        self.episode_reward += reward
        self.done = done

        self.memory.store({'observations': observation, 'done': done, 'rewards': reward})

        self.optimiser.run()

        action_values = self.exploration.bias_action_value(observation)
        action = self.action_fa.convert(action_values)

        self.memory.store({'actions': action})

        if done:
            self.exploration.update()

        return action
