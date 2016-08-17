from rlagents.agents import RandomAgent, EvolutionaryAgent, ExploratoryAgent
from rlagents.models import TabularModel, DiscreteActionLinearModel, ContinuousActionLinearModel
from rlagents.function_approximation import SingleTiling
from rlagents.optimisation.evolutionary import CrossEntropy, GeneticAlgorithm, SimulatedAnnealing, HillClimbing
from rlagents.memory import PandasMemory


def random(action_space, observation_space):
    agent = RandomAgent(action_space, observation_space)
    return agent


def crossentropy_tabular(action_space, observation_space):
    model = TabularModel(action_space.n, SingleTiling(observation_space, 8))
    agent = EvolutionaryAgent(action_space, observation_space, model=model, evolution=CrossEntropy(), batch_size=40)
    return agent


def crossentropy_discretelinear(action_space, observation_space):
    model = DiscreteActionLinearModel(action_space, observation_space)
    agent = EvolutionaryAgent(action_space, observation_space, model=model, evolution=CrossEntropy(), batch_size=40)
    return agent


def geneticalgorithm_discretelinear(action_space, observation_space):
    model = DiscreteActionLinearModel(action_space, observation_space)
    agent = EvolutionaryAgent(action_space, observation_space, model=model, evolution=GeneticAlgorithm(), batch_size=40)
    return agent


def simulatedannealing_discretelinear(action_space, observation_space):
    model = DiscreteActionLinearModel(action_space, observation_space)
    agent = EvolutionaryAgent(action_space, observation_space, model=model, evolution=SimulatedAnnealing(), batch_size=1)
    return agent


def hillclimbing_discretelinear(action_space, observation_space):
    model = DiscreteActionLinearModel(action_space, observation_space)
    agent = EvolutionaryAgent(action_space, observation_space, model=model, evolution=HillClimbing(), batch_size=40)
    return agent


def standard(action_space, observation_space):
    agent = ExploratoryAgent(action_space, observation_space)
    return agent


def standard_pandasmemory(action_space, observation_space):
    pm = PandasMemory(size=1, columns=['observations', 'actions'])
    agent = ExploratoryAgent(action_space, observation_space, memory=pm)
    return agent


def crossentropy_continuouslinear(action_space, observation_space):
    model = ContinuousActionLinearModel(action_space, observation_space)
    agent = EvolutionaryAgent(action_space, observation_space, model=model, evolution=CrossEntropy(), batch_size=40)
    return agent


def geneticalgorithm_continuouslinear(action_space, observation_space):
    model = ContinuousActionLinearModel(action_space, observation_space)
    agent = EvolutionaryAgent(action_space, observation_space, model=model, evolution=GeneticAlgorithm(), batch_size=40)
    return agent


def simulatedannealing_continuouslinear(action_space, observation_space):
    model = ContinuousActionLinearModel(action_space, observation_space)
    agent = EvolutionaryAgent(action_space, observation_space, model=model, evolution=SimulatedAnnealing(), batch_size=1)
    return agent


def hillclimbing_continuouslinear(action_space, observation_space):
    model = ContinuousActionLinearModel(action_space, observation_space)
    agent = EvolutionaryAgent(action_space, observation_space, model=model, evolution=HillClimbing(), batch_size=40)
    return agent

