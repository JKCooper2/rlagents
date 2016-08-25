import warnings

from rlagents.function_approximation import DefaultFA, SingleTiling, FunctionApproximationBase, DiscreteMaxFA
from rlagents.models import ModelBase, TabularModel
from rlagents.memory import ListMemory
from rlagents.optimisation import DefaultOptimiser, OptimiserBase
from rlagents.exploration import DefaultExploration, ExplorationBase


class Agent(object):
    def __init__(self, action_space, observation_space, action_fa=None, model=None, exploration=None, memory=None, optimiser=None):
        self.action_space = action_space
        self.observation_space = observation_space

        self.action_fa = action_fa
        self.model = model

        self.memory = memory
        self.action_fa = action_fa

        self.exploration = exploration
        self.optimiser = optimiser

        self.episode_reward = 0
        self.done = False

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

    @property
    def exploration(self):
        return self._exploration

    @exploration.setter
    def exploration(self, ex):
        if not isinstance(ex, ExplorationBase):
            ex = DefaultExploration()
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
            o = DefaultOptimiser()
            warnings.warn("Optimiser is not a valid OptimiserBase")

        self._optimiser = o

        # Probably not the best idea to require ordering of variable loading but simpler than
        # passing through variables each time
        self._optimiser.model = self.model
        self._optimiser.memory = self.memory

    def next_agent(self):
        """Keeps functionality in EnvManager consistent with Pools"""
        self.episode_reward = 0
        return self

    def act(self, observation, reward, done):
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
