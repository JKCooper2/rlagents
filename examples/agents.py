from rlagents.agents import Agent
from rlagents.models import WeightedLinearModel, DefaultModel
from rlagents.function_approximation import ClipFA, DiscreteMaxFA
from rlagents.evolution import CrossEntropy, GeneticAlgorithm, SimulatedAnnealing, HillClimbing
from rlagents.exploration import RandomExploration
from rlagents.pool import Pool


def random_discrete():
    agent = Agent(model=DefaultModel(),
                  exploration=RandomExploration(),
                  action_fa=DiscreteMaxFA())
    return agent


def random_continuous():
    agent = Agent(model=DefaultModel(),
                  exploration=RandomExploration(),
                  action_fa=ClipFA())

    return agent


def crossentropy_discretelinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=DiscreteMaxFA())
    pool = Pool(CrossEntropy())
    pool.add(agent, number=2)
    return pool


def geneticalgorithm_discretelinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=DiscreteMaxFA())
    pool = Pool(GeneticAlgorithm())
    pool.add(agent, number=2)
    return pool


def simulatedannealing_discretelinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=DiscreteMaxFA())
    pool = Pool(SimulatedAnnealing())
    pool.add(agent, number=1)
    return pool


def hillclimbing_discretelinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=DiscreteMaxFA())
    pool = Pool(HillClimbing())
    pool.add(agent, number=2)
    return pool


def crossentropy_continuouslinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=ClipFA())
    pool = Pool(CrossEntropy())
    pool.add(agent, number=2)
    return pool


def geneticalgorithm_continuouslinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=ClipFA())
    pool = Pool(GeneticAlgorithm())
    pool.add(agent, number=2)
    return pool


def simulatedannealing_continuouslinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=ClipFA())
    pool = Pool(SimulatedAnnealing())
    pool.add(agent, number=1)
    return pool


def hillclimbing_continuouslinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=ClipFA())
    pool = Pool(HillClimbing())
    pool.add(agent, number=2)
    return pool

