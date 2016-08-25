from rlagents.agents import Agent
from rlagents.models import TabularModel, WeightedLinearModel
from rlagents.function_approximation import SingleTiling, DefaultFA, ClipFA, DiscreteMaxFA
from rlagents.evolution import CrossEntropy, GeneticAlgorithm, SimulatedAnnealing, HillClimbing
from rlagents.memory import PandasMemory
from rlagents.pool import Pool


def crossentropy_tabular(action_space, observation_space):
    model = TabularModel(action_space.n, SingleTiling(observation_space, 8))
    agent = Agent(action_space, observation_space, model=model)
    pool = Pool(CrossEntropy(), times_run=1)
    pool.add(agent, number=2)
    return pool


def crossentropy_discretelinear(action_space, observation_space):
    model = WeightedLinearModel(DiscreteMaxFA(action_space), DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model)
    pool = Pool(CrossEntropy(), times_run=1)
    pool.add(agent, number=2)
    return pool


def geneticalgorithm_discretelinear(action_space, observation_space):
    model = WeightedLinearModel(DiscreteMaxFA(action_space), DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model)
    pool = Pool(GeneticAlgorithm(), times_run=1)
    pool.add(agent, number=2)
    return pool


def simulatedannealing_discretelinear(action_space, observation_space):
    model = WeightedLinearModel(DiscreteMaxFA(action_space), DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model)
    pool = Pool(SimulatedAnnealing(), times_run=1)
    pool.add(agent, number=1)
    return pool


def hillclimbing_discretelinear(action_space, observation_space):
    model = WeightedLinearModel(DiscreteMaxFA(action_space), DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, model=model)
    pool = Pool(HillClimbing(), times_run=1)
    pool.add(agent, number=2)
    return pool


def standard(action_space, observation_space):
    agent = Agent(action_space, observation_space)
    return agent


def standard_pandasmemory(action_space, observation_space):
    pm = PandasMemory(size=1, columns=['observations', 'actions'])
    agent = Agent(action_space, observation_space, memory=pm)
    return agent


def crossentropy_continuouslinear(action_space, observation_space):
    model = WeightedLinearModel(ClipFA(action_space), DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, action_fa=ClipFA(action_space), model=model)
    pool = Pool(CrossEntropy(), times_run=1)
    pool.add(agent, number=2)
    return pool


def geneticalgorithm_continuouslinear(action_space, observation_space):
    model = WeightedLinearModel(ClipFA(action_space), DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, action_fa=ClipFA(action_space), model=model)
    pool = Pool(GeneticAlgorithm(), times_run=1)
    pool.add(agent, number=2)
    return pool


def simulatedannealing_continuouslinear(action_space, observation_space):
    model = WeightedLinearModel(ClipFA(action_space), DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, action_fa=ClipFA(action_space), model=model)
    pool = Pool(SimulatedAnnealing(), times_run=1)
    pool.add(agent, number=1)
    return pool


def hillclimbing_continuouslinear(action_space, observation_space):
    model = WeightedLinearModel(ClipFA(action_space), DefaultFA(observation_space))
    agent = Agent(action_space, observation_space, action_fa=ClipFA(action_space), model=model)
    pool = Pool(HillClimbing(), times_run=1)
    pool.add(agent, number=2)
    return pool

