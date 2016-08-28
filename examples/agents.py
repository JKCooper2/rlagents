from rlagents.agents import Agent
from rlagents.models import WeightedLinearModel, DefaultModel
from rlagents.function_approximation import ClipFA, DiscreteMaxFA
from rlagents.evolution import CrossEntropy, GeneticAlgorithm, SimulatedAnnealing, HillClimbing
from rlagents.exploration import RandomExploration
from rlagents.agent_manager import AgentManager


def random_discrete():
    agent = Agent(model=DefaultModel(),
                  exploration=RandomExploration(),
                  action_fa=DiscreteMaxFA())
    am = AgentManager(agent=agent)
    return am


def random_continuous():
    agent = Agent(model=DefaultModel(),
                  exploration=RandomExploration(),
                  action_fa=ClipFA())
    am = AgentManager(agent=agent)
    return am

def random_default():
    agent = Agent(model=DefaultModel(),
                  exploration=RandomExploration())
    am = AgentManager(agent)
    return am


def crossentropy_discretelinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=DiscreteMaxFA())
    am = AgentManager(CrossEntropy())
    am.add(agent, number=2)
    return am


def geneticalgorithm_discretelinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=DiscreteMaxFA())
    am = AgentManager(GeneticAlgorithm())
    am.add(agent, number=2)
    return am


def simulatedannealing_discretelinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=DiscreteMaxFA())
    am = AgentManager(SimulatedAnnealing())
    am.add(agent, number=1)
    return am


def hillclimbing_discretelinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=DiscreteMaxFA())
    am = AgentManager(HillClimbing())
    am.add(agent, number=2)
    return am


def crossentropy_continuouslinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=ClipFA())
    am = AgentManager(CrossEntropy())
    am.add(agent, number=2)
    return am


def geneticalgorithm_continuouslinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=ClipFA())
    am = AgentManager(GeneticAlgorithm())
    am.add(agent, number=2)
    return am


def simulatedannealing_continuouslinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=ClipFA())
    am = AgentManager(SimulatedAnnealing())
    am.add(agent, number=1)
    return am


def hillclimbing_continuouslinear():
    agent = Agent(model=WeightedLinearModel(),
                  action_fa=ClipFA())
    am = AgentManager(HillClimbing())
    am.add(agent, number=2)
    return am

